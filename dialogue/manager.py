"""Confidence-routed dialogue: answer, clarify, hand to the AI, or escalate.

Order of checks per turn matters: exit and pending clarification first (they
depend on the previous turn), then small talk and order lookups (cheap and
exact), then FAQ retrieval, and the LLM only when retrieval isn't confident.

Every turn is handled in the language the customer used (English, Telugu or
Hindi, detected per message), and every reply comes from that language's pack
in data/i18n/, so wording can be fixed without touching code.
"""

import re
import time
from collections import Counter
from dataclasses import dataclass

import config
from nlu.entities import OrderLookup, extract_order_id
from nlu.language import LanguagePack, detect_language, load_pack, normalize_any
from nlu.matcher import FAQMatcher, content_words, normalize
from nlu.sentiment import SentimentAnalyzer

EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye", "bye bye", "stop", "end", "that is all", "thats all"}

YES = {"yes", "yeah", "yep", "yup", "yes please", "sure", "correct", "right", "exactly", "ya", "haan", "ok", "okay"}
NO = {"no", "nope", "nah", "not really", "wrong", "no thanks", "neither"}

FOLLOWUP_WORDS = {"that", "it", "this", "those", "them", "same"}
FOLLOWUP_PREFIXES = ("and ", "what about", "how about", "also ")
ORDER_FAQS = {"order_status", "delayed_order", "cancel_order", "modify_order"}

# Topics that commonly follow each other, used to break ties on follow-ups
# like "how long does that take?" after asking about returns.
RELATED_CATEGORIES = {
    "returns": {"returns", "refunds"},
    "refunds": {"refunds", "returns"},
    "orders": {"orders", "shipping"},
    "shipping": {"shipping", "orders"},
}
FOLLOWUP_BOOST = 0.12

_EN = load_pack("en")
ESCALATE_MSG = _EN.msg("escalate")
HANDOFF_MSG = _EN.msg("handoff")
GOODBYE_MSG = _EN.msg("goodbye")
PRIORITY_MSG = _EN.msg("priority")

# English small talk keeps regexes for flexible phrasing; Telugu/Hindi use
# the phrase lists in their language packs.
SMALL_TALK = [
    (re.compile(r"^(hi|hello|hey|hii+|good (morning|afternoon|evening)|namaste)( there)?( aria)?$"), "hello"),
    (re.compile(r"^(thanks|thank you|thank you so much|thanks a lot|thx|ty)( aria)?$"), "thanks"),
    (re.compile(r"^(ok|okay|cool|great|got it|alright|fine|perfect|nice)$"), "ack"),
    (re.compile(r"^how are you( doing)?( today)?$"), "how_are_you"),
    (re.compile(r"^(who are you|what are you|what is your name|are you (a )?(bot|robot|human|real))$"), "who_are_you"),
    (re.compile(r"^(what can you do|help|what do you do)$"), "capabilities"),
]
SMALL_TALK_KEYS = ("hello", "thanks", "ack", "how_are_you", "who_are_you", "capabilities")
_INDIC_MATCHERS: dict = {}
_DEFAULT_MATCHER: list = []


def _default_matcher() -> FAQMatcher:
    # Built once per process and shared, for the same reason as the Indic ones.
    if not _DEFAULT_MATCHER:
        _DEFAULT_MATCHER.append(FAQMatcher())
    return _DEFAULT_MATCHER[0]


HUMAN_REQUEST = re.compile(r"\b(human|agent|representative|real person|talk to (someone|a person)|speak to (someone|a person)|live chat)\b")


@dataclass
class Turn:
    query: str
    response: str
    action: str             # answer | order | ai | clarify | escalate | smalltalk | exit
    confidence: float = 0.0
    source: str = ""        # faq id, "order", "ai:<provider>", ...
    latency_ms: float = 0.0
    sentiment: str = "neutral"
    lang: str = "en"

    @property
    def is_exit(self) -> bool:
        return self.action == "exit"


class DialogueManager:
    def __init__(self, matcher: FAQMatcher | None = None, ai=None, orders: OrderLookup | None = None,
                 language: str = "en"):
        self.matcher = matcher or _default_matcher()  # English matcher; owns the canonical FAQ list
        self.ai = ai            # LLMAssistant or None for rule-only
        self.orders = orders or OrderLookup()
        self.sentiment = SentimentAnalyzer()
        self.lang = language
        self._frustration = 0    # builds with angry/negative turns, decays when calm
        self._pending = None     # (Match, original query) awaiting yes/no
        self._last_faq = None    # most recent FAQ answered, for follow-ups
        self._history: list[dict] = []
        self._turns: list[Turn] = []

    # ------------------------------------------------------------------ public
    def handle(self, query: str) -> Turn:
        start = time.perf_counter()
        query = query.strip()
        self.lang = detect_language(query, self.lang)
        pack = load_pack(self.lang)
        mood = self.sentiment.analyze(query, pack)
        turn = self._route(query, pack)
        if not turn.is_exit:
            turn = self._apply_sentiment(turn, mood.label, pack)
        turn.sentiment = mood.label
        turn.lang = self.lang
        turn.latency_ms = (time.perf_counter() - start) * 1000
        if not turn.is_exit:
            self._turns.append(turn)
            self._history += [{"role": "user", "content": turn.query},
                              {"role": "assistant", "content": turn.response}]
            self._history = self._history[-2 * config.AI_HISTORY_TURNS:]
        return turn

    def greeting(self) -> str:
        return load_pack(self.lang).msg("greeting")

    def stats(self) -> dict:
        n = len(self._turns)
        count = lambda *actions: sum(t.action in actions for t in self._turns)
        return {
            "turns": n,
            "answered": count("answer", "order", "ai"),
            "ai_answered": count("ai"),
            "clarified": count("clarify"),
            "escalated": count("escalate"),
            "upset_turns": sum(t.sentiment in ("negative", "angry") for t in self._turns),
            "languages": dict(Counter(t.lang for t in self._turns)),
            "avg_latency_ms": round(sum(t.latency_ms for t in self._turns) / n, 2) if n else 0.0,
        }

    def matcher_for(self, lang: str):
        if lang == "en":
            return self.matcher
        # Indexes are read-only, so they're shared by every conversation in
        # the process: a new customer session shouldn't rebuild them.
        key = (lang, id(self.matcher.faqs))
        if key not in _INDIC_MATCHERS:
            from nlu.indic_matcher import IndicMatcher
            _INDIC_MATCHERS[key] = IndicMatcher(load_pack(lang), self.matcher.faqs)
        return _INDIC_MATCHERS[key]

    # --------------------------------------------------------------- sentiment
    def _apply_sentiment(self, turn: Turn, mood: str, pack: LanguagePack) -> Turn:
        self._frustration = max(0, self._frustration + {"angry": 2, "negative": 1}.get(mood, -1))
        if self._frustration >= config.FRUSTRATION_ESCALATE:
            # Repeated anger means the bot isn't helping, however good its
            # answers look; a human with the context is the better service.
            self._frustration = 0
            self._pending = None
            return Turn(turn.query, pack.msg("priority"), "escalate", 1.0, "sentiment")
        apologetic = _EN.apologetic + pack.apologetic
        if (mood in ("angry", "negative") and turn.action in ("answer", "order", "ai", "clarify")
                and not turn.response.lower().startswith(apologetic)):
            turn.response = f"{pack.msg('empathy_' + mood)} {turn.response}"
        return turn

    # ----------------------------------------------------------------- routing
    def _route(self, query: str, pack: LanguagePack) -> Turn:
        english = pack.code == "en"
        norm = normalize(query) if english else normalize_any(query)
        if not norm:
            return Turn(query, pack.msg("not_heard"), "clarify")

        if norm in EXIT_COMMANDS or pack.is_word("exit", norm):
            return Turn(query, pack.msg("goodbye"), "exit")

        if self._pending is not None:
            pending, original = self._pending
            self._pending = None
            if norm in YES or norm.startswith("yes ") or pack.is_word("yes", norm):
                return self._answer(query, pending.faq, pending.score, pack)
            if norm in NO or norm.startswith("no ") or pack.is_word("no", norm):
                return self._fallback(original, pack, rejected=True)
            # Anything else is treated as a fresh question.
        elif norm in NO or pack.is_word("no", norm):
            # A bare "no" with nothing pending ("anything else?" - "no") is
            # an answer, not a question to match against the FAQ.
            return Turn(query, pack.msg("no_ack"), "smalltalk", 1.0, "smalltalk")
        elif norm in YES or pack.is_word("yes", norm):
            return Turn(query, pack.msg("ack"), "smalltalk", 1.0, "smalltalk")

        key = self._small_talk(norm, pack)
        if key:
            return Turn(query, pack.msg(key), "smalltalk", 1.0, "smalltalk")
        if HUMAN_REQUEST.search(normalize(query)) or pack.contains_word("human", norm):
            return Turn(query, pack.msg("handoff"), "escalate", 1.0, "handoff")

        order_turn = self._try_order(query, norm, pack)
        if order_turn:
            return order_turn

        if english:
            matches = self._match_with_context(query, norm)
        else:
            matches = self.matcher_for(pack.code).match(query, top_k=5)
        top = matches[0] if matches else None

        if top and top.score >= config.ANSWER_THRESHOLD:
            return self._answer(query, top.faq, top.score, pack)
        if top and top.score >= config.CLARIFY_THRESHOLD:
            self._pending = (top, query)
            question = self._localized(top.faq, "question", pack)
            return Turn(query, pack.msg("clarify", question=question), "clarify", top.score, top.id)
        return self._fallback(query, pack, score=top.score if top else 0.0)

    @staticmethod
    def _small_talk(norm: str, pack: LanguagePack) -> str | None:
        if pack.code == "en":
            for pattern, key in SMALL_TALK:
                if pattern.match(norm):
                    return key
            return None
        for key in SMALL_TALK_KEYS:
            if pack.is_word(key, norm):
                return key
        return None

    @staticmethod
    def _localized(faq: dict, field: str, pack: LanguagePack) -> str:
        return pack.faq.get(faq["id"], {}).get(field) or faq[field]

    def _answer(self, query, faq, score, pack: LanguagePack) -> Turn:
        self._last_faq = faq
        return Turn(query, self._localized(faq, "answer", pack), "answer", score, faq["id"])

    def _fallback(self, query: str, pack: LanguagePack, score: float = 0.0, rejected: bool = False) -> Turn:
        """Low confidence: let the AI try with the full FAQ as its knowledge
        base; escalate if it's unavailable or says the FAQ doesn't cover it."""
        if self.ai is not None and self.ai.available:
            knowledge, relevant = self._ranked_knowledge(query, pack.code)
            reply, provider = self.ai.answer(query, knowledge, self._history,
                                             language=pack.name, relevant=relevant)
            if reply:
                return Turn(query, reply, "ai", score, f"ai:{provider}")
            return Turn(query, pack.msg("escalate"), "escalate", score, f"ai:{provider}" if provider else "")
        if rejected:
            return Turn(query, pack.msg("rephrase"), "clarify", score)
        return Turn(query, pack.msg("escalate"), "escalate", score)

    def _ranked_knowledge(self, query: str, lang: str) -> tuple[list[dict], int]:
        """All FAQs, most relevant to this question first, plus how many of
        them actually matched. A local model only reads the top few."""
        matched = [m.faq for m in self.matcher_for(lang).match(query, top_k=len(self.matcher.faqs))]
        seen = {f["id"] for f in matched}
        return matched + [f for f in self.matcher.faqs if f["id"] not in seen], len(matched)

    def _try_order(self, query: str, norm: str, pack: LanguagePack) -> Turn | None:
        order_id = extract_order_id(query)
        if not order_id:
            # A bare number that isn't a valid order id (e.g. "1234" typed in
            # reply to "what's your order number?") deserves a hint, not a
            # handoff to a human.
            digits = norm.replace(" ", "")
            if digits.isdigit():
                return Turn(query, pack.msg("bad_order_number", digits=digits), "clarify", 0.0, "order")
            return None
        # A bare 5-8 digit number only counts as an order id when the user
        # mentioned an order or we were just talking about orders; otherwise
        # "50000 was deducted twice" would be looked up as an order.
        order_context = ("order" in norm or pack.contains_word("order", norm)
                         or any(w in norm for w in pack.words.get("order", ()))
                         or norm.replace(" ", "").isdigit()
                         or (self._last_faq is not None and self._last_faq["id"] in ORDER_FAQS))
        if not order_context:
            return None
        order = self.orders.orders.get(order_id)
        if order is None:
            return Turn(query, pack.msg("order_not_found", id=order_id), "clarify", 0.0, "order")
        text = pack.msg("order_found", id=order_id, item=order["item"],
                        status=pack.status.get(order["status"], order["status"]),
                        detail=pack.orders.get(order_id, order["detail"]))
        return Turn(query, text, "order", 1.0, "order")

    def _match_with_context(self, query: str, norm: str):
        matches = self.matcher.match(query, top_k=5)
        if self._last_faq is None or not self._is_followup(norm):
            return matches
        related = RELATED_CATEGORIES.get(self._last_faq["category"], {self._last_faq["category"]})
        for m in matches:
            if m.faq["category"] in related:
                m.score += FOLLOWUP_BOOST
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches

    @staticmethod
    def _is_followup(norm: str) -> bool:
        words = norm.split()
        return (len(content_words(norm)) <= 3
                and (bool(FOLLOWUP_WORDS & set(words)) or norm.startswith(FOLLOWUP_PREFIXES)))
