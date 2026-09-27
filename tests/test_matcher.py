"""Rule-engine tests. No network or models: the AI fallback is replaced by a
fake provider so routing logic is verified deterministically."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import config
from ai.llm import LLMAssistant
from dialogue.manager import DialogueManager, EXIT_COMMANDS
from nlu.entities import extract_order_id
from nlu.matcher import FAQMatcher, normalize


@pytest.fixture(scope="module")
def matcher():
    return FAQMatcher()


def top(matcher, q):
    m = matcher.match(q)
    return (m[0].id, m[0].score) if m else (None, 0.0)


# --- matcher -----------------------------------------------------------------
@pytest.mark.parametrize("query,expected", [
    ("How do I reset my password?", "reset_password"),
    ("What is your return policy?", "return_policy"),
    ("How long does delivery take?", "shipping_time"),
    ("What payment methods do you accept?", "payment_methods"),
])
def test_exact_phrasing(matcher, query, expected):
    faq_id, score = top(matcher, query)
    assert faq_id == expected
    assert score >= config.ANSWER_THRESHOLD


@pytest.mark.parametrize("query,expected", [
    ("i forgot my pasword", "reset_password"),
    ("how do i cancle my ordr", "cancel_order"),
    ("when will i get my refnd", "refund_time"),
    ("is cash on delivary available", "cod"),
])
def test_typos(matcher, query, expected):
    assert top(matcher, query)[0] == expected


@pytest.mark.parametrize("query,expected", [
    ("money got deducted but the payment failed", "payment_failed"),
    ("the product I got is broken", "damaged_item"),
    ("do you deliver to other countries", "international_shipping"),
    ("how can I talk to customer care", "contact_support"),
])
def test_paraphrase(matcher, query, expected):
    assert top(matcher, query)[0] == expected


def test_contractions_normalized():
    assert normalize("I can't log in") == "i cannot log in"
    assert normalize("It won't load") == "it will not load"
    assert normalize("I didn’t get it") == "i did not get it"  # curly apostrophe


@pytest.mark.parametrize("query", [
    "what is the weather in delhi today",
    "who won the cricket match yesterday",
    "tell me a recipe for biryani",
    "what is",
])
def test_off_topic_rejected(matcher, query):
    assert top(matcher, query)[1] < config.CLARIFY_THRESHOLD


# --- entities ------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("where is order 10234", "10234"),
    ("my order number is #10567", "10567"),
    ("1 0 8 9 1", "10891"),
    ("how do I return an item", None),
])
def test_order_id_extraction(text, expected):
    assert extract_order_id(text) == expected


# --- dialogue ------------------------------------------------------------------
@pytest.fixture
def bot():
    return DialogueManager(ai=None)


def test_confident_answer(bot):
    t = bot.handle("how do I reset my password")
    assert t.action == "answer" and t.source == "reset_password"


def test_clarify_then_confirm(bot, matcher):
    # Find a query that lands in the clarify band, rather than hard-coding a
    # score that shifts whenever the FAQ data is edited.
    query = "my package was left open"
    faq_id, score = top(matcher, query)
    assert config.CLARIFY_THRESHOLD <= score < config.ANSWER_THRESHOLD, score

    t = bot.handle(query)
    assert t.action == "clarify"
    t = bot.handle("yes")
    assert t.action == "answer" and t.source == faq_id


def test_clarify_then_reject(bot):
    bot.handle("my package was left open")
    t = bot.handle("no")
    assert t.action == "clarify" and "rephrase" in t.response


def test_off_topic_escalates_without_ai(bot):
    assert bot.handle("what is the weather in delhi").action == "escalate"


def test_order_lookup(bot):
    t = bot.handle("where is my order 10567")
    assert t.action == "order" and "Running Shoes" in t.response
    assert bot.handle("check order 99999").action == "clarify"


def test_short_bare_number_gets_hint_not_escalation(bot):
    bot.handle("track order")
    t = bot.handle("1234")
    assert t.action == "clarify" and "5 digits" in t.response
    assert bot.handle("10234").action == "order"


def test_large_number_without_order_context_is_not_an_order(bot):
    assert bot.handle("50000 rupees payment failed").source != "order"


def test_followup_uses_context(bot):
    bot.handle("how do I return an item")
    t = bot.handle("how long does that take")
    assert t.source == "refund_time"


def test_smalltalk_and_handoff(bot):
    assert bot.handle("hello").action == "smalltalk"
    assert bot.handle("thanks").action == "smalltalk"
    assert bot.handle("I want to talk to a human").action == "escalate"


@pytest.mark.parametrize("cmd", sorted(EXIT_COMMANDS))
def test_exit_commands(bot, cmd):
    assert bot.handle(cmd).is_exit


def test_stats(bot):
    bot.handle("how do I reset my password")
    bot.handle("what is the weather in delhi")
    bot.handle("exit")
    s = bot.stats()
    assert s["turns"] == 2 and s["answered"] == 1 and s["escalated"] == 1
    assert s["avg_latency_ms"] >= 0


# --- AI fallback (fake providers, no network) ---------------------------------
class FakeProvider:
    def __init__(self, name, reply=None, error=None):
        self.name, self.key, self.reply, self.error = name, "fake", reply, error
        self.calls = []

    def generate(self, system, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.reply


def test_ai_answers_low_confidence():
    fake = FakeProvider("fake", "Our stores are online only, so we don't have a physical location.")
    bot = DialogueManager(ai=LLMAssistant([fake]))
    t = bot.handle("do you have a shop in hyderabad")
    assert t.action == "ai" and t.source == "ai:fake"
    # The model must be grounded on the FAQ knowledge base.
    assert "Knowledge base" in fake.calls[0][-1]["content"]


def test_ai_not_called_for_confident_match():
    fake = FakeProvider("fake", "should not be used")
    bot = DialogueManager(ai=LLMAssistant([fake]))
    assert bot.handle("how do I reset my password").action == "answer"
    assert fake.calls == []


def test_ai_escalate_token_escalates():
    bot = DialogueManager(ai=LLMAssistant([FakeProvider("fake", "ESCALATE")]))
    assert bot.handle("what is the weather in delhi").action == "escalate"


def test_ai_falls_back_to_next_provider():
    broken = FakeProvider("broken", error=ConnectionError("offline"))
    working = FakeProvider("working", "Here's the answer.")
    bot = DialogueManager(ai=LLMAssistant([broken, working]))
    t = bot.handle("do you have a shop in hyderabad")
    assert t.action == "ai" and t.source == "ai:working"


def test_all_providers_failing_escalates():
    bot = DialogueManager(ai=LLMAssistant([FakeProvider("a", error=TimeoutError()),
                                           FakeProvider("b", error=TimeoutError())]))
    assert bot.handle("do you have a shop in hyderabad").action == "escalate"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
