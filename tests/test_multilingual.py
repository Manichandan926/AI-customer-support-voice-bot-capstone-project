"""Telugu and Hindi: language detection, matching, localized replies, order
lookup, sentiment, and pack completeness. Offline - no ASR/TTS/network."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import config
from dialogue.manager import DialogueManager
from nlu.language import SUPPORTED, detect_language, load_pack, normalize_any


# --- language packs are complete -------------------------------------------------
@pytest.mark.parametrize("lang", ["te", "hi"])
def test_pack_covers_every_faq_message_and_order(lang):
    pack = load_pack(lang)
    faq_ids = {f["id"] for f in json.load(open(config.FAQ_PATH, encoding="utf-8"))}
    orders = set(json.load(open(config.ORDERS_PATH, encoding="utf-8")))
    assert set(pack.faq) == faq_ids
    assert set(load_pack("en").messages) <= set(pack.messages)
    assert orders <= set(pack.orders)
    for entry in pack.faq.values():
        assert entry["question"] and entry["answer"] and entry["variants"]


# --- detection --------------------------------------------------------------------
@pytest.mark.parametrize("text,lang", [
    ("where is my order", "en"),
    ("నా ఆర్డర్ ఎక్కడ ఉంది", "te"),
    ("मेरा ऑर्डर कहाँ है", "hi"),
    ("నా order ఎక్కడ ఉంది", "te"),          # code-mixed
    ("naa order ekkada undi", "te"),        # romanized Telugu
    ("mera order kahan hai", "hi"),         # romanized Hindi
    ("my payment failed", "en"),
])
def test_detect_language(text, lang):
    assert detect_language(text) == lang


def test_short_or_numeric_replies_keep_conversation_language():
    assert detect_language("10234", current="te") == "te"
    assert detect_language("ok", current="hi") == "hi"


def test_spelling_variants_fold_together():
    assert normalize_any("రీఫండ్") == normalize_any("రిఫండ్")
    assert normalize_any("कहाँ") == normalize_any("कहां") == normalize_any("कहा")
    assert normalize_any("ऑर्डर") == normalize_any("आर्डर")


# --- matching and localized replies ------------------------------------------------
@pytest.mark.parametrize("lang,query,faq_id", [
    ("te", "నా ఆర్డర్ ఎక్కడ ఉంది", "order_status"),
    ("te", "నా ఆర్డర్‌ను ఎలా క్యాన్సిల్ చేయాలి", "cancel_order"),
    ("te", "రీఫండ్ ఇంకా రాలేదు", "refund_time"),
    ("te", "పాస్‌వర్డ్ మర్చిపోయాను", "reset_password"),
    ("te", "naa order ekkada undi", "order_status"),
    ("te", "నా రిఫండ్ ఎప్పుడు వస్తుంది", "refund_time"),      # ASR spelling variant
    ("hi", "मेरा ऑर्डर कहाँ है", "order_status"),
    ("hi", "मेरा आर्डर कहा है", "order_status"),               # casual spelling
    ("hi", "मेरा रिफंड अभी तक नहीं आया", "refund_time"),
    ("hi", "पैसे कट गए लेकिन पेमेंट फेल", "payment_failed"),
    ("hi", "mera refund kab aayega", "refund_time"),
])
def test_indic_questions_answered_in_their_language(lang, query, faq_id):
    t = DialogueManager(ai=None).handle(query)
    assert t.lang == lang
    assert t.action == "answer" and t.source == faq_id
    assert t.response == load_pack(lang).faq[faq_id]["answer"]


@pytest.mark.parametrize("query", ["హైదరాబాద్‌లో వాతావరణం ఎలా ఉంది", "दिल्ली में मौसम कैसा है"])
def test_indic_off_topic_escalates_in_same_language(query):
    t = DialogueManager(ai=None).handle(query)
    assert t.action == "escalate"
    assert t.response == load_pack(t.lang).msg("escalate")


def test_language_switches_mid_conversation():
    bot = DialogueManager(ai=None)
    assert bot.handle("how do I reset my password").lang == "en"
    assert bot.handle("నా ఆర్డర్ ఎక్కడ ఉంది").lang == "te"
    assert bot.handle("मेरा ऑर्डर कहाँ है").lang == "hi"
    assert bot.stats()["languages"] == {"en": 1, "te": 1, "hi": 1}


@pytest.mark.parametrize("lang,hello", [("te", "నమస్కారం"), ("hi", "नमस्ते")])
def test_indic_small_talk(lang, hello):
    t = DialogueManager(ai=None).handle(hello)
    assert t.action == "smalltalk" and t.response == load_pack(lang).msg("hello")


def test_telugu_order_lookup_is_localized():
    bot = DialogueManager(ai=None)
    bot.handle("నా ఆర్డర్ ఎక్కడ ఉంది")
    t = bot.handle("10234")
    pack = load_pack("te")
    assert t.action == "order" and t.lang == "te"
    assert pack.status["shipped"] in t.response and pack.orders["10234"] in t.response


def test_hindi_bare_no_is_acknowledged_not_matched():
    t = DialogueManager(ai=None, language="hi").handle("नहीं")
    assert t.action == "smalltalk" and t.response == load_pack("hi").msg("no_ack")


def test_telugu_exit():
    assert DialogueManager(ai=None).handle("బై").is_exit


# --- sentiment in Telugu/Hindi -----------------------------------------------------
def test_hindi_anger_escalates_with_priority():
    bot = DialogueManager(ai=None)
    bot.handle("बकवास सर्विस है, बेकार")   # two anger words -> angry (+2)
    t = bot.handle("घटिया, मेरा रिफंड नहीं आया")  # +1 -> threshold
    assert t.action == "escalate" and t.response == load_pack("hi").msg("priority")


def test_telugu_complaint_gets_localized_empathy():
    t = DialogueManager(ai=None).handle("చెత్త సర్వీస్, రీఫండ్ ఇంకా రాలేదు")
    assert t.sentiment == "negative"
    assert t.response.startswith(load_pack("te").msg("empathy_negative"))


# --- AI fallback is asked to reply in the customer's language ---------------------
def test_ai_prompt_requests_customer_language():
    from ai.llm import LLMAssistant

    class Fake:
        name, key = "fake", "k"
        def __init__(self): self.calls = []
        def generate(self, system, messages):
            self.calls.append(messages)
            return "సమాధానం"

    fake = Fake()
    t = DialogueManager(ai=LLMAssistant([fake])).handle("మీకు హైదరాబాద్‌లో షాప్ ఉందా")
    assert t.action == "ai"
    assert "Reply in Telugu" in fake.calls[0][-1]["content"]


def test_supported_languages():
    assert SUPPORTED == ("en", "te", "hi")
