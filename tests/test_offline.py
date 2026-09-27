"""Online/offline switching for speech recognition and voices. No network,
microphone, speaker or downloaded models: the engines are replaced with
fakes so only the switching logic is under test."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

import config
import network
from asr import offline_asr
from asr.speech_to_text import SpeechToText
from tts import offline_tts
from tts.text_to_speech import TextToSpeech

AUDIO = np.zeros(1600, dtype=np.int16)


@pytest.fixture(autouse=True)
def fresh_network(monkeypatch):
    monkeypatch.setattr(config, "OFFLINE_MODE", "auto")
    network.reset()
    yield
    network.reset()


def set_online(monkeypatch, online: bool):
    calls = []

    def fake_connect(*args, **kwargs):
        calls.append(args)
        if not online:
            raise OSError("no route")
        class Conn:
            def close(self): pass
        return Conn()

    monkeypatch.setattr(network.socket, "create_connection", fake_connect)
    return calls


# --- connectivity ------------------------------------------------------------------
def test_offline_mode_always_never_touches_network(monkeypatch):
    calls = set_online(monkeypatch, True)
    monkeypatch.setattr(config, "OFFLINE_MODE", "always")
    assert network.is_online() is False and calls == []


def test_connectivity_result_is_cached(monkeypatch):
    calls = set_online(monkeypatch, True)
    assert network.is_online() and network.is_online() and network.is_online()
    assert len(calls) == 1


def test_mark_offline_skips_network_until_ttl(monkeypatch):
    calls = set_online(monkeypatch, True)
    network.mark_offline()
    assert network.is_online() is False and calls == []


# --- speech recognition --------------------------------------------------------------
def test_online_uses_google(monkeypatch):
    set_online(monkeypatch, True)
    stt = SpeechToText(lang="en")
    monkeypatch.setattr(stt, "_recognize_google", lambda s: "where is my order")
    assert stt.transcribe_samples(AUDIO) == "where is my order" and stt.last_engine == "google"


def test_google_failure_falls_back_to_vosk(monkeypatch):
    set_online(monkeypatch, True)
    stt = SpeechToText(lang="hi")

    def google_down(samples):
        raise ConnectionError("request failed")

    monkeypatch.setattr(stt, "_recognize_google", google_down)
    monkeypatch.setattr(offline_asr, "is_available", lambda lang: True)
    monkeypatch.setattr(offline_asr.VoskRecognizer, "transcribe", lambda self, s: "मेरा ऑर्डर कहाँ है")
    assert stt.transcribe_samples(AUDIO) == "मेरा ऑर्डर कहाँ है" and stt.last_engine == "vosk"
    assert network.is_online() is False  # the next turn skips Google straight away


def test_offline_without_model_explains_how_to_install(monkeypatch):
    set_online(monkeypatch, False)
    monkeypatch.setattr(offline_asr, "is_available", lambda lang: False)
    with pytest.raises(RuntimeError, match="tools.download_models --lang te"):
        SpeechToText(lang="te").transcribe_samples(AUDIO)


def test_online_only_mode_does_not_fall_back(monkeypatch):
    set_online(monkeypatch, False)
    monkeypatch.setattr(config, "OFFLINE_MODE", "never")
    with pytest.raises(RuntimeError, match="unreachable"):
        SpeechToText(lang="en").transcribe_samples(AUDIO)


def test_telugu_grammar_is_native_script_vocabulary():
    grammar = offline_asr.grammar_for("te")
    assert grammar[-1] == "[unk]"
    assert "ఆర్డర్" in grammar and "నా ఆర్డర్ ఎక్కడ ఉంది" in grammar
    assert not any(g.isascii() for g in grammar[:-1])  # romanized variants excluded


# --- voices ----------------------------------------------------------------------------
@pytest.fixture
def tts(monkeypatch):
    t = TextToSpeech()
    spoken = []
    monkeypatch.setattr(t, "_speak_edge", lambda text, voice: spoken.append(("edge", voice)))
    monkeypatch.setattr(t, "_speak_sapi", lambda text: spoken.append(("system", None)))
    monkeypatch.setattr(offline_tts.PiperSpeaker, "speak", lambda self, text, lang: spoken.append(("piper", lang)))
    t.spoken = spoken
    return t


def test_online_speaks_with_neural_voice(monkeypatch, tts):
    set_online(monkeypatch, True)
    tts.speak("నమస్కారం", voice="te-IN-ShrutiNeural", lang="te")
    assert tts.spoken == [("edge", "te-IN-ShrutiNeural")] and tts.last_engine == "edge"


def test_offline_uses_piper_when_installed(monkeypatch, tts):
    set_online(monkeypatch, False)
    monkeypatch.setattr(offline_tts, "is_available", lambda lang: True)
    tts.speak("नमस्ते", lang="hi")
    assert tts.spoken == [("piper", "hi")] and tts.last_engine == "piper"


def test_edge_failure_falls_back_to_piper_and_marks_offline(monkeypatch, tts):
    set_online(monkeypatch, True)
    monkeypatch.setattr(offline_tts, "is_available", lambda lang: True)

    def edge_down(text, voice):
        raise OSError("connection reset")

    monkeypatch.setattr(tts, "_speak_edge", edge_down)
    tts.speak("hello", lang="en")
    assert tts.spoken == [("piper", "en")] and network.is_online() is False


def test_offline_english_without_piper_uses_system_voice(monkeypatch, tts):
    set_online(monkeypatch, False)
    monkeypatch.setattr(offline_tts, "is_available", lambda lang: False)
    tts.speak("hello", lang="en")
    assert tts.spoken == [("system", None)]


def test_offline_telugu_without_piper_is_text_only(monkeypatch, tts):
    set_online(monkeypatch, False)
    monkeypatch.setattr(offline_tts, "is_available", lambda lang: False)
    with pytest.raises(RuntimeError, match="download_models --lang te"):
        tts.speak("నమస్కారం", lang="te")
    assert tts.spoken == []


# --- model downloader ------------------------------------------------------------------
def test_piper_download_urls():
    from tools.download_models import piper_url
    assert piper_url("te_IN-maya-medium", ".onnx") == (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/te/te_IN/maya/medium/te_IN-maya-medium.onnx")
