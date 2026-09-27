"""Confidence-routed dialogue: answer, clarify, hand to the AI, or escalate.

Order of checks per turn matters: exit and pending clarification first (they
depend on the previous turn), then small talk and order lookups (cheap and
exact), then FAQ retrieval, and the LLM only when retrieval isn't confident.
"""

import re
import time
from dataclasses import dataclass

import config
from nlu.entities import OrderLookup, extract_order_id
from nlu.matcher import FAQMatcher, content_words, normalize

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

ESCALATE_MSG = ("I'm not sure about that one, so I'm connecting you to a human agent. "
                "You can also call us toll-free at 1800-123-4567.")
HANDOFF_MSG = "Sure, I'm transferring you to a human agent now. Please stay on the line."
GOODBYE_MSG = f"Thanks for contacting {config.COMPANY_NAME}. Have a great day!"

SMALL_TALK = [
    (re.compile(r"^(hi|hello|hey|hii+|good (morning|afternoon|evening)|namaste)( there)?( aria)?$"),
     f"Hello! I'm {config.BOT_NAME}, your {config.COMPANY_NAME} assistant. How can I help you today?"),
    (re.compile(r"^(thanks|thank you|thank you so much|thanks a lot|thx|ty)( aria)?$"),
     "You're welcome! Is there anything else I can help you with?"),
    (re.compile(r"^(ok|okay|cool|great|got it|alright|fine|perfect|nice)$"),
     "Great! Is there anything else I can help you with?"),
    (re.compile(r"^how are you( doing)?( today)?$"),
     "I'm doing great, thanks for asking! What can I help you with?"),
    (re.compile(r"^(who are you|what are you|what is your name|are you (a )?(bot|robot|human|real))$"),
     f"I'm {config.BOT_NAME}, an AI support assistant for {config.COMPANY_NAME}. I can help with orders, "
     "returns, refunds, payments, shipping and your account."),
    (re.compile(r"^(what can you do|help|what do you do)$"),
     "I can track orders, explain returns and refunds, help with payments and billing, "
     "shipping questions, and account issues. Just ask!"),
]
HUMAN_REQUEST = re.compile(r"\b(human|agent|representative|real person|talk to (someone|a person)|speak to (someone|a person)|live chat)\b")


@dataclass
class Turn:
    query: str
    response: str
    action: str             # answer | order | ai | clarify | escalate | smalltalk | exit
    confidence: float = 0.0
    source: str = ""        # faq id, "order", "ai:<provider>", ...
    latency_ms: float = 0.0

    @property
    def is_exit(self) -> bool:
        return self.action == "exit"


class DialogueManager:
    def __init__(self, matcher: FAQMatcher | None = None, ai=None, orders: OrderLookup | None = None):
        self.matcher = matcher or FAQMatcher()
        self.ai = ai            # LLMAssistant or None for rule-only
        self.orders = orders or OrderLookup()
        self._pending = None     # (Match, original query) awaiting yes/no
        self._last_faq = None    # most recent FAQ answered, for follow-ups
        self._history: list[dict] = []
        self._turns: list[Turn] = []

    # ------------------------------------------------------------------ public
    def handle(self, query: str) -> Turn:
        start = time.perf_counter()
        turn = self._route(query.strip())
        turn.latency_ms = (time.perf_counter() - start) * 1000
        if not turn.is_exit:
            self._turns.append(turn)
            self._history += [{"role": "user", "content": turn.query},
                              {"role": "assistant", "content": turn.response}]
            self._history = self._history[-2 * config.AI_HISTORY_TURNS:]
        return turn

    def stats(self) -> dict:
        n = len(self._turns)
        count = lambda *actions: sum(t.action in actions for t in self._turns)
        return {
            "turns": n,
            "answered": count("answer", "order", "ai"),
            "ai_answered": count("ai"),
            "clarified": count("clarify"),
            "escalated": count("escalate"),
            "avg_latency_ms": round(sum(t.latency_ms for t in self._turns) / n, 2) if n else 0.0,
        }

    # ----------------------------------------------------------------- routing
    def _route(self, query: str) -> Turn:
        norm = normalize(query)
        if not norm:
            return Turn(query, "Sorry, I didn't catch that. Could you say it again?", "clarify")

        if norm in EXIT_COMMANDS:
            return Turn(query, GOODBYE_MSG, "exit")

        if self._pending is not None:
            pending, original = self._pending
            self._pending = None
            if norm in YES or norm.startswith("yes "):
                return self._answer(query, pending.faq, pending.score)
            if norm in NO or norm.startswith("no "):
                return self._fallback(original, "No problem. Could you rephrase your question, "
                                                "or say 'agent' to talk to a person?")
            # Anything else is treated as a fresh question.

        for pattern, reply in SMALL_TALK:
            if pattern.match(norm):
                return Turn(query, reply, "smalltalk", 1.0, "smalltalk")
        if HUMAN_REQUEST.search(norm):
            return Turn(query, HANDOFF_MSG, "escalate", 1.0, "handoff")

        order_turn = self._try_order(query, norm)
        if order_turn:
            return order_turn

        matches = self._match_with_context(query, norm)
        top = matches[0] if matches else None

        if top and top.score >= config.ANSWER_THRESHOLD:
            return self._answer(query, top.faq, top.score)
        if top and top.score >= config.CLARIFY_THRESHOLD:
            self._pending = (top, query)
            return Turn(query, f"Just to confirm, are you asking: \"{top.faq['question']}\"?",
                        "clarify", top.score, top.id)
        return self._fallback(query, ESCALATE_MSG, top.score if top else 0.0)

    def _answer(self, query, faq, score) -> Turn:
        self._last_faq = faq
        return Turn(query, faq["answer"], "answer", score, faq["id"])

    def _fallback(self, query: str, no_ai_reply: str, score: float = 0.0) -> Turn:
        """Low confidence: let the AI try with the full FAQ as its knowledge
        base; escalate if it's unavailable or says the FAQ doesn't cover it."""
        if self.ai is not None and self.ai.available:
            reply, provider = self.ai.answer(query, self.matcher.faqs, self._history)
            if reply:
                return Turn(query, reply, "ai", score, f"ai:{provider}")
            return Turn(query, ESCALATE_MSG, "escalate", score, f"ai:{provider}" if provider else "")
        action = "escalate" if no_ai_reply == ESCALATE_MSG else "clarify"
        return Turn(query, no_ai_reply, action, score)

    def _try_order(self, query: str, norm: str) -> Turn | None:
        order_id = extract_order_id(query)
        if not order_id:
            # A bare number that isn't a valid order id (e.g. "1234" typed in
            # reply to "what's your order number?") deserves a hint, not a
            # handoff to a human.
            digits = norm.replace(" ", "")
            if digits.isdigit():
                return Turn(query, f"{digits} doesn't look like an order number. Order numbers are "
                                   "5 digits, like 10234 - you'll find yours in the confirmation email.",
                            "clarify", 0.0, "order")
            return None
        # A bare 5-8 digit number only counts as an order id when the user
        # mentioned an order or we were just talking about orders; otherwise
        # "50000 was deducted twice" would be looked up as an order.
        order_context = ("order" in norm or norm.replace(" ", "").isdigit()
                         or (self._last_faq is not None and self._last_faq["id"] in ORDER_FAQS))
        if not order_context:
            return None
        desc = self.orders.describe(order_id)
        if desc:
            return Turn(query, desc, "order", 1.0, "order")
        return Turn(query, f"I couldn't find an order with number {order_id}. Please check the number "
                           "in your confirmation email and try again.", "clarify", 0.0, "order")

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
