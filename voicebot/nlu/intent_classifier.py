"""
nlu/intent_classifier.py

Stands in for the NLU (intent classification) stage of the pipeline.
Right now it's simple keyword matching. Later, this is the file you
replace with a real classifier (e.g. a fine-tuned DistilBERT via ONNX,
as planned for the fuller pipeline) — the rest of the system only
depends on the Intent object this returns, not on how it was produced.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Intent:
    """Result of intent classification for one user query."""
    label: str          # e.g. "account_issue"
    confidence: float    # 0.0-1.0 — real models produce this; ours is a stand-in
    matched_keywords: List[str]


class KeywordIntentClassifier:
    """Very simple rule-based intent matcher.

    Each intent has a list of trigger keywords. The input query is
    lowercased and checked against each intent's keyword list. The
    intent with the most keyword hits wins. If nothing matches, the
    query falls back to "general_inquiry".
    """

    # Keyword table — easy to extend without touching any logic below.
    # Longer phrases are intentionally checked before shorter ones so the
    # classifier prefers specific user intents over generic words.
    INTENT_KEYWORDS: Dict[str, List[str]] = {
        "greeting": [
            "hello", "hi there", "good morning", "good afternoon",
            "good evening", "hey there", "howdy", "greetings", "sup",
            "hello there",
        ],
        "account_issue": [
            "forgot my password", "reset my password", "can't log in",
            "cannot log in", "account locked", "locked out", "unlock my account",
            "sign in issue", "change password", "login issue", "password reset",
            "access my account", "profile problem", "username issue", "account",
            "password", "login", "log in", "sign in", "reset", "username", "profile",
        ],
        "technical_support": [
            "not working", "error", "bug", "crash", "broken",
            "install", "setup", "device", "connect", "cannot connect",
            "can't connect", "wifi", "bluetooth", "app keeps crashing",
            "connection problem", "technical issue", "system error",
        ],
        "billing_inquiry": [
            "charged twice", "double charge", "invoice", "payment failed",
            "refund", "subscription", "price", "cost", "plan",
            "billing", "charge", "charged", "refund request", "overcharged",
            "bill", "payment",
        ],
        "order_status": [
            "track my package", "track my order", "where is my package",
            "where is my order", "when will my order arrive",
            "when will my package arrive", "my package hasn't arrived",
            "my delivery is late", "shipment", "delivery", "tracking",
            "shipped", "package", "order", "arrive",
        ],
    }

    FALLBACK_INTENT = "general_inquiry"

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.lower().split())

    def classify(self, text: str) -> Intent:
        text_lower = self._normalize_text(text)

        best_label = self.FALLBACK_INTENT
        best_matches: List[str] = []

        for label, keywords in self.INTENT_KEYWORDS.items():
            matches = []
            for kw in keywords:
                if kw in text_lower:
                    matches.append(kw)
            if len(matches) > len(best_matches):
                best_matches = matches
                best_label = label

        if not best_matches:
            return Intent(label=self.FALLBACK_INTENT, confidence=0.3, matched_keywords=[])

        confidence = min(0.45 + 0.12 * len(best_matches), 0.95)
        if best_label == self.FALLBACK_INTENT:
            confidence = 0.3

        return Intent(label=best_label, confidence=confidence, matched_keywords=best_matches)
