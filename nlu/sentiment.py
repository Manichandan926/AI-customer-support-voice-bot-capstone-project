"""Offline sentiment: VADER polarity plus a customer-support anger lexicon.

VADER alone can't separate a normal complaint ("the item arrived broken",
which deserves an apology and the FAQ answer) from real anger ("useless
bot, this is a scam", which should reach a human sooner). The anger lexicon
and shouting cues make that distinction. Both are rule-based, so this costs
microseconds and no model.
"""

import re
from dataclasses import dataclass

import config

ANGER_TERMS = [
    "worst", "useless", "pathetic", "ridiculous", "disgusting", "horrible", "terrible",
    "fraud", "scam", "cheat", "cheated", "cheating", "liar", "lying", "rubbish", "nonsense",
    "angry", "furious", "frustrated", "frustrating", "fed up", "sick of", "had enough",
    "waste of time", "waste of money", "never again", "never buy", "consumer court",
    "legal action", "sue you", "no one is helping", "nobody is helping", "nobody helps",
    "not helping", "stupid", "hate", "annoyed", "annoying", "unacceptable", "shameful",
]
_ANGER_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in ANGER_TERMS) + r")\b", re.I)


@dataclass
class Sentiment:
    label: str        # positive | neutral | negative | angry
    score: float      # VADER compound, -1..1
    anger_hits: int = 0


class SentimentAnalyzer:
    # Shared across instances: the VADER lexicon takes ~10 ms to load and
    # never changes, so each new conversation shouldn't reload it.
    _vader = None

    @classmethod
    def _load_vader(cls):
        if cls._vader is None:
            try:
                import warnings
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                with warnings.catch_warnings():
                    # VADER's loader uses codecs.open, deprecated on Python 3.14.
                    warnings.simplefilter("ignore", DeprecationWarning)
                    cls._vader = SentimentIntensityAnalyzer()
            except ImportError:
                cls._vader = False  # lexicon-only mode still catches anger
        return cls._vader

    def _polarity(self, text: str) -> float:
        vader = self._load_vader()
        return vader.polarity_scores(text)["compound"] if vader else 0.0

    def analyze(self, text: str) -> Sentiment:
        score = self._polarity(text)
        hits = len(_ANGER_RE.findall(text))
        # Shouting: several all-caps words or repeated "!!" / "??".
        shouting = sum(1 for w in text.split() if len(w) > 2 and w.isupper()) >= 2
        if shouting or re.search(r"[!?]{2,}", text):
            hits += 1
        if hits and score <= config.ANGRY_SCORE:
            label = "angry"
        elif score <= config.NEGATIVE_SCORE or hits:
            label = "negative"
        elif score >= config.POSITIVE_SCORE:
            label = "positive"
        else:
            label = "neutral"
        return Sentiment(label, score, hits)
