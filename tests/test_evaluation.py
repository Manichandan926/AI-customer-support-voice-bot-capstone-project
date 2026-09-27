"""Accuracy regression guard: runs the held-out evaluation set and fails if
quality drops below today's measured level (minus a small margin), so a
careless edit to the FAQ data or the matcher can't silently make the bot
worse. Measured at the time of writing: resolved 88%, precision 89%,
off-topic rejection 92%, language detection 100%."""

import pytest

from tools.evaluate import run


@pytest.fixture(scope="module")
def summary():
    return run()["summary"]


def test_resolves_most_in_scope_questions(summary):
    assert summary["all"]["resolved_incl_clarify"] >= 0.85


def test_rarely_answers_the_wrong_question(summary):
    assert summary["all"]["answer_precision"] >= 0.85


def test_rejects_off_topic_questions(summary):
    assert summary["all"]["offtopic_rejected"] >= 0.85


def test_detects_every_language(summary):
    assert summary["all"]["language_detected"] == 1.0


@pytest.mark.parametrize("lang", ["te", "hi"])
def test_indic_languages_never_answer_wrongly(summary, lang):
    assert summary[lang]["wrong"] == 0
