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
    INTENT_KEYWORDS: Dict[str, List[str]] = {
        "greeting": [
            "hi", "hello", "hey", "good morning", "good afternoon",
            "good evening", "howdy", "greetings", "sup",
        ],
        "account_issue": [
            "account", "password", "login", "log in", "sign in",
            "locked out", "reset", "username", "profile",
        ],
        "technical_support": [
            "not working", "error", "bug", "crash", "broken",
            "install", "setup", "device", "connect", "wifi", "bluetooth",
        ],
        "billing_inquiry": [
            "bill", "invoice", "payment", "charge", "refund",
            "subscription", "price", "cost", "plan",
        ],
        "order_status": [
            "order", "shipment", "delivery", "tracking", "shipped",
            "package", "arrive",
        ],
    }

    FALLBACK_INTENT = "general_inquiry"

    def classify(self, text: str) -> Intent:
        text_lower = text.lower()

        best_label = self.FALLBACK_INTENT
        best_matches: List[str] = []

        for label, keywords in self.INTENT_KEYWORDS.items():
            matches = [kw for kw in keywords if kw in text_lower]
            if len(matches) > len(best_matches):
                best_matches = matches
                best_label = label

        if not best_matches:
            return Intent(label=self.FALLBACK_INTENT, confidence=0.3, matched_keywords=[])

        # Toy confidence score: more keyword hits -> higher confidence.
        # A real model would output a proper probability distribution.
        confidence = min(0.5 + 0.15 * len(best_matches), 0.95)
        return Intent(label=best_label, confidence=confidence, matched_keywords=best_matches)
