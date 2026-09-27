"""Offline neural voices with Piper (ONNX Runtime on CPU, ~63 MB per voice).
Sounds far more natural than the built-in Windows/espeak voices, and has
Telugu and Hindi voices, which those don't.
"""

import config

_voices = {}  # loaded once per language


def voice_path(lang: str):
    name = config.PIPER_VOICES.get(lang)
    return config.MODELS_DIR / "piper" / f"{name}.onnx" if name else None


def is_available(lang: str) -> bool:
    path = voice_path(lang)
    return path is not None and path.is_file() and path.with_suffix(".onnx.json").is_file()


class PiperSpeaker:
    def _voice(self, lang: str):
        if lang not in _voices:
            try:
                from piper import PiperVoice
            except ImportError:
                raise RuntimeError("piper-tts isn't installed, run: pip install piper-tts") from None
            if not is_available(lang):
                raise RuntimeError(f"no offline voice for '{lang}', run: python -m tools.download_models --lang {lang}")
            # CPU inference only; Piper's optional CUDA path is deliberately not used.
            _voices[lang] = PiperVoice.load(voice_path(lang))
        return _voices[lang]

    def synthesize(self, text: str, lang: str):
        """Returns (int16 samples, sample_rate)."""
        import numpy as np
        from piper import SynthesisConfig
        voice = self._voice(lang)
        chunks = list(voice.synthesize(text, SynthesisConfig(length_scale=config.PIPER_LENGTH_SCALE)))
        if not chunks:
            return np.zeros(0, dtype=np.int16), config.SAMPLE_RATE
        audio = np.concatenate([c.audio_int16_array for c in chunks])
        return audio, chunks[0].sample_rate

    def speak(self, text: str, lang: str) -> None:
        import sounddevice as sd
        audio, rate = self.synthesize(text, lang)
        if len(audio):
            sd.play(audio, rate)
            sd.wait()
