"""Sentiment, end-of-speech detection and cross-platform playback. No mic,
speaker or network needed: audio is simulated as per-frame RMS levels and
the player is a stubbed subprocess call."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import config
from asr.speech_to_text import FRAME_SECONDS, Endpointer
from dialogue.manager import PRIORITY_MSG, DialogueManager
from nlu.sentiment import SentimentAnalyzer


# --- sentiment -----------------------------------------------------------------
@pytest.fixture(scope="module")
def analyzer():
    return SentimentAnalyzer()


@pytest.mark.parametrize("text,label", [
    ("how do I reset my password", "neutral"),
    ("thank you so much, that was helpful", "positive"),
    ("the product arrived broken", "negative"),
    ("this is the worst service, useless bot", "angry"),
    ("WHERE IS MY REFUND this is ridiculous!!", "angry"),
    ("i am so frustrated, nobody is helping me", "angry"),
])
def test_sentiment_labels(analyzer, text, label):
    assert analyzer.analyze(text).label == label


def test_empathy_prefix_on_negative():
    bot = DialogueManager(ai=None)
    t = bot.handle("my payment failed and i am upset")
    assert t.sentiment in ("negative", "angry")
    assert t.response.lower().startswith(("i'm sorry", "i'm really sorry", "don't worry"))


def test_no_double_apology():
    t = DialogueManager(ai=None).handle("the product arrived broken")
    assert t.response.lower().count("sorry") == 1


def test_repeated_anger_escalates_with_priority():
    bot = DialogueManager(ai=None)
    bot.handle("the product arrived broken")
    t = bot.handle("this is the worst service ever, useless")
    assert t.action == "escalate" and t.response == PRIORITY_MSG and t.source == "sentiment"


def test_frustration_decays_when_calm():
    bot = DialogueManager(ai=None)
    bot.handle("the product arrived broken")        # +1
    bot.handle("how do I reset my password")       # -1
    bot.handle("how do I cancel my order")         # stays 0
    t = bot.handle("my refund is late and i am upset")  # +1, below threshold
    assert t.action != "escalate" or t.source != "sentiment"


def test_stats_count_upset_turns():
    bot = DialogueManager(ai=None)
    bot.handle("the product arrived broken")
    bot.handle("how do I reset my password")
    assert bot.stats()["upset_turns"] == 1


# --- end-of-speech detection ------------------------------------------------------
def frames(seconds):
    return int(round(seconds / FRAME_SECONDS))


def run(levels, noise=50.0):
    ep = Endpointer(noise_floor=noise)
    for i, lvl in enumerate(levels):
        state = ep.feed(lvl)
        if state in ("done", "timeout"):
            return state, i + 1
    return "unfinished", len(levels)


def test_endpointer_stops_after_trailing_silence():
    speech = [2000] * frames(1.5)
    silence = [40] * frames(3)
    state, n = run([40] * frames(0.5) + speech + silence)
    assert state == "done"
    stopped_after = (n - frames(2.0)) * FRAME_SECONDS  # time past end of speech
    assert stopped_after == pytest.approx(config.VAD_END_SILENCE, abs=0.1)


def test_endpointer_times_out_when_nobody_speaks():
    state, n = run([40] * frames(config.VAD_START_TIMEOUT + 1))
    assert state == "timeout"


def test_endpointer_ignores_single_click():
    state, _ = run([40] * 10 + [5000] + [40] * frames(config.VAD_START_TIMEOUT))
    assert state == "timeout"


def test_endpointer_keeps_short_pauses_inside_speech():
    pause = [40] * frames(config.VAD_END_SILENCE / 2)
    levels = [2000] * frames(1) + pause + [2000] * frames(1) + [40] * frames(2)
    ep = Endpointer(noise_floor=50)
    states = [ep.feed(l) for l in levels]
    first_done = states.index("done")
    assert first_done > frames(2)  # did not stop during the mid-sentence pause


def test_endpointer_threshold_adapts_to_noisy_room():
    # In a loud room (floor 800), 1500 is background, not speech.
    state, _ = run([1500] * frames(config.VAD_START_TIMEOUT + 1), noise=800)
    assert state == "timeout"


def test_endpointer_caps_long_utterances():
    state, n = run([2000] * frames(config.VAD_MAX_SECONDS + 5))
    assert state == "done" and n * FRAME_SECONDS <= config.VAD_MAX_SECONDS + 0.5


# --- cross-platform playback -------------------------------------------------------
def test_linux_uses_first_available_player(monkeypatch):
    import tts.text_to_speech as t
    calls = []
    monkeypatch.setattr(t.os, "name", "posix")
    monkeypatch.setattr(t.shutil, "which", lambda cmd: "/usr/bin/ffplay" if cmd == "ffplay" else None)
    monkeypatch.setattr(t.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    t._play_mp3("/tmp/x.mp3")
    assert calls and calls[0][0] == "ffplay" and calls[0][-1] == "/tmp/x.mp3"


def test_linux_without_player_gives_actionable_error(monkeypatch):
    import tts.text_to_speech as t
    monkeypatch.setattr(t.os, "name", "posix")
    monkeypatch.setattr(t.shutil, "which", lambda cmd: None)
    with pytest.raises(RuntimeError, match="apt install mpg123"):
        t._play_mp3("/tmp/x.mp3")


def test_noise_level_ignores_warmup_zeros_and_uses_upper_quartile():
    from asr.speech_to_text import noise_level
    levels = [0, 0, 0, 200, 210, 220, 230, 600]  # zeros = mic waking up, 600 = a burst
    assert noise_level(levels) == 230
    assert noise_level([0, 0]) == 0.0
