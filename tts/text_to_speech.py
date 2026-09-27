"""Text-to-speech in Aria's voice, with an offline chain behind it:

1. Online: Microsoft's neural voices via edge-tts (free, no API key) -
   Neerja / Shruti / Swara for English / Telugu / Hindi.
2. Offline: Piper neural voices (downloaded once, run on the CPU), in all
   three languages.
3. Last resort, English only: the operating system's voice via pyttsx3
   (Windows Zira, Linux espeak-ng), which needs no download at all.

The online voice is skipped whenever the network is known to be down, so an
outage costs one short connectivity check, not a timeout on every reply.
"""

import asyncio
import ctypes
import os
import shutil
import subprocess
import sys
import tempfile

import config
import network


class TextToSpeech:
    def __init__(self):
        self._edge_ok = config.TTS_ENGINE == "edge"
        self._pyttsx3 = None
        self._piper = None
        self.last_engine = ""

    def speak(self, text: str, voice: str | None = None, lang: str = "en") -> None:
        if self._edge_ok and network.is_online():
            try:
                self._speak_edge(text, voice or config.TTS_VOICE)
                self.last_engine = "edge"
                return
            except ImportError as e:
                self._edge_ok = False  # not installed: won't fix itself this session
                print(f"  [tts] {e}", file=sys.stderr)
            except Exception as e:
                # Most likely the network dropped; recheck later rather than
                # giving up on the nicer voice for the rest of the session.
                network.mark_offline()
                print(f"  [tts] online voice failed ({type(e).__name__}); using offline voice", file=sys.stderr)
        if config.OFFLINE_MODE == "never":
            raise RuntimeError("online voice unavailable - reply shown as text only")

        from tts.offline_tts import PiperSpeaker, is_available
        if is_available(lang):
            if self._piper is None:
                self._piper = PiperSpeaker()
            self._piper.speak(text, lang)
            self.last_engine = "piper"
            return
        if lang != "en":
            # The built-in Windows/espeak voices can't pronounce Telugu or
            # Hindi script, so say so instead of reading out garbage.
            raise RuntimeError(f"no offline voice for this language - reply shown as text only. "
                               f"Install one with: python -m tools.download_models --lang {lang}")
        self._speak_sapi(text)
        self.last_engine = "system"

    # ------------------------------------------------------------- neural voice
    def _speak_edge(self, text: str, voice_name: str) -> None:
        try:
            import edge_tts
        except ImportError:
            raise ImportError("edge-tts isn't installed, run: pip install edge-tts") from None
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
