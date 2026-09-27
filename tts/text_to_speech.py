"""Text-to-speech in Aria's voice.

Primary: Microsoft's neural voices via edge-tts (free, no API key, no local
model - synthesis happens on Microsoft's servers, like Google ASR does for
input). It sounds far smoother than the SAPI5 voices built into Windows.
Fallback: pyttsx3 with Windows' offline female voice (Zira), used whenever
edge-tts isn't installed or there's no internet, so voice output never
just stops working mid-demo.
"""

import asyncio
import ctypes
import os
import shutil
import subprocess
import sys
import tempfile

import config


class TextToSpeech:
    def __init__(self):
        self._edge_ok = config.TTS_ENGINE == "edge"
        self._pyttsx3 = None

    def speak(self, text: str, voice: str | None = None, lang: str = "en") -> None:
        if self._edge_ok:
            try:
                self._speak_edge(text, voice or config.TTS_VOICE)
                return
            except Exception as e:
                # Don't retry the neural voice every turn once it has failed;
                # a flaky connection would add seconds of delay to each reply.
                self._edge_ok = False
                print(f"  [tts] neural voice unavailable ({type(e).__name__}: {str(e)[:80]}); "
                      "using offline Windows voice", file=sys.stderr)
        if lang != "en":
            # The offline Windows/espeak voices can't pronounce Telugu or
            # Hindi script, so say so instead of reading out garbage.
            raise RuntimeError("the offline voice can't speak this language - reply shown as text only")
        self._speak_sapi(text)

    # ------------------------------------------------------------- neural voice
    def _speak_edge(self, text: str, voice_name: str) -> None:
        try:
            import edge_tts
        except ImportError:
            raise RuntimeError("edge-tts isn't installed, run: pip install edge-tts") from None
        fd, path = tempfile.mkstemp(suffix=".mp3", prefix="aria_")
        os.close(fd)
        try:
            voice = edge_tts.Communicate(text, voice_name, rate=config.TTS_NEURAL_RATE,
                                         pitch=config.TTS_NEURAL_PITCH)
            asyncio.run(voice.save(path))
            _play_mp3(path)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    # ------------------------------------------------------------ offline voice
    def _speak_sapi(self, text: str) -> None:
        if self._pyttsx3 is None:
            try:
                import pyttsx3
            except ImportError:
                raise RuntimeError("pyttsx3 isn't installed, run: pip install pyttsx3") from None
            self._pyttsx3 = pyttsx3
        # A fresh engine per utterance: reusing one across runAndWait() calls
        # silently produces no audio on some Windows builds.
        try:
            engine = self._pyttsx3.init()
        except Exception as e:
            raise RuntimeError(f"Text-to-speech failed to start ({e}). If this is a COM error, run: "
                               r"python venv\Scripts\pywin32_postinstall.py -install") from None
        for v in engine.getProperty("voices"):
            if config.TTS_OFFLINE_VOICE.lower() in v.name.lower():
                engine.setProperty("voice", v.id)
                break
        else:
            if os.name != "nt":
                # Zira is Windows-only; espeak-ng's "+f3" variant is its
                # clearest female voice.
                try:
                    engine.setProperty("voice", "en+f3")
                except Exception:
                    pass
        engine.setProperty("rate", config.TTS_RATE)
        engine.setProperty("volume", config.TTS_VOLUME)
        engine.say(text)
        engine.runAndWait()
        engine.stop()


# Command-line players tried on Linux/macOS, lightest first. mpg123 is a
# ~200 KB apt package and the recommended choice on a Raspberry Pi.
_PLAYERS = [
    ["mpg123", "-q"],
    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
    ["mpv", "--no-video", "--really-quiet"],
    ["cvlc", "--play-and-exit", "--quiet"],
    ["afplay"],  # macOS built-in
]


def _play_mp3(path: str) -> None:
    if os.name == "nt":
        _play_mp3_windows(path)
        return
    for cmd in _PLAYERS:
        if shutil.which(cmd[0]):
            subprocess.run([*cmd, path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
    raise RuntimeError("no mp3 player found, run: sudo apt install mpg123")


def _play_mp3_windows(path: str) -> None:
    """Plays an mp3 through Windows' built-in MCI player, so no audio
    library or codec needs installing."""
    winmm = ctypes.windll.winmm

    def mci(cmd: str) -> None:
        err = winmm.mciSendStringW(cmd, None, 0, None)
        if err:
            buf = ctypes.create_unicode_buffer(256)
            winmm.mciGetErrorStringW(err, buf, 256)
            raise RuntimeError(f"MCI: {buf.value}")

    mci(f'open "{path}" type mpegvideo alias aria')
    try:
        mci("play aria wait")
    finally:
        winmm.mciSendStringW("close aria", None, 0, None)
