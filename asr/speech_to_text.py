"""Speech-to-text via the SpeechRecognition library's free Google recognizer.

Audio is captured with sounddevice rather than SpeechRecognition's own
Microphone class, because that needs PyAudio, which often has no prebuilt
wheel for new Python versions on Windows.
"""

import config


class SpeechToText:
    def __init__(self, language: str = config.ASR_LANGUAGE):
        self.language = language
        self._sr = None
        self._recognizer = None

    def _load(self):
        if self._recognizer is None:
            try:
                import speech_recognition as sr
            except ImportError:
                raise RuntimeError("SpeechRecognition isn't installed, run: pip install SpeechRecognition") from None
            self._sr = sr
            self._recognizer = sr.Recognizer()
        return self._sr

    def record(self, seconds: float = config.RECORD_SECONDS):
        """Records from the default microphone; returns int16 samples, or None if silent."""
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError:
            raise RuntimeError("Microphone support isn't installed, run: pip install sounddevice numpy") from None
        try:
            audio = sd.rec(int(seconds * config.SAMPLE_RATE), samplerate=config.SAMPLE_RATE,
                           channels=1, dtype="int16")
            sd.wait()
        except Exception as e:
            raise RuntimeError(f"Couldn't record from the microphone ({e}). "
                               "Check that a mic is connected and allowed in Windows privacy settings.") from None
        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        return None if rms < config.SILENCE_RMS else audio

    def transcribe_samples(self, samples) -> str:
        sr = self._load()
        data = sr.AudioData(samples.tobytes(), config.SAMPLE_RATE, 2)
        return self._recognize(data)

    def transcribe_file(self, path: str) -> str:
        sr = self._load()
        with sr.AudioFile(path) as source:
            data = self._recognizer.record(source)
        return self._recognize(data)

    def listen(self, seconds: float = config.RECORD_SECONDS) -> str:
        samples = self.record(seconds)
        return "" if samples is None else self.transcribe_samples(samples)

    def _recognize(self, data) -> str:
        try:
            return self._recognizer.recognize_google(data, language=self.language)
        except self._sr.UnknownValueError:
            return ""  # speech was unintelligible
        except self._sr.RequestError as e:
            raise RuntimeError(f"Google speech service unreachable ({e}). Check your internet connection, "
                               "or type your question instead.") from None
