"""FAQ retrieval: normalize -> stem -> spell-correct -> TF-IDF + fuzzy score.

Each query word is stemmed ("delivered" -> "deliver") and, if unknown,
snapped to the closest word in the FAQ vocabulary ("pasword" -> "password").
Word TF-IDF then ranks FAQs by meaning (rare words like "refund" count for
more than common ones like "order"), and a whole-string fuzzy score rewards
reordered phrasing. A content-word gate on top means generic overlaps
("how do I ...", "what is ...") never produce a match on their own.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config

try:
    from rapidfuzz import fuzz, process

    def _sorted_ratio(a: str, b: str) -> float:
        return fuzz.token_sort_ratio(a, b)

    def _closest(word: str, vocab: list[str]) -> str | None:
        hit = process.extractOne(word, vocab, scorer=fuzz.ratio, score_cutoff=config.SPELL_MATCH)
        return hit[0] if hit else None
except ImportError:  # rapidfuzz is faster, but difflib keeps things working without it
    from difflib import SequenceMatcher, get_close_matches

    def _sorted_ratio(a: str, b: str) -> float:
        a, b = " ".join(sorted(a.split())), " ".join(sorted(b.split()))
        return SequenceMatcher(None, a, b).ratio() * 100

    def _closest(word: str, vocab: list[str]) -> str | None:
        hits = get_close_matches(word, vocab, n=1, cutoff=config.SPELL_MATCH / 100)
        return hits[0] if hits else None


CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "shan't": "shall not",
    "n't": " not", "'re": " are", "'m": " am", "'ve": " have",
    "'ll": " will", "'d": " would", "'s": "",
}

# Words that appear in almost every support question and say nothing about
# the topic. Treating them as stop words is what stops "what is the weather"
# from matching "what is your return policy".
GENERIC_WORDS = {
    "please", "help", "want", "need", "know", "tell", "like", "get", "got",
    "would", "could", "can", "hi", "hello", "hey", "thanks", "thank", "ok",
    "okay", "yes", "i", "im", "my", "me", "you", "your", "shopease",
    "way", "able", "possible", "just", "really", "actually", "also", "does",
    "did", "doing", "dont", "didnt", "hasnt", "havent", "isnt", "wasnt",
}
# sklearn's list drops words that change meaning in support queries
# ("log out" vs "log in", "money back", "not working").
MEANINGFUL = {"not", "cannot", "no", "out", "back", "off", "down", "up", "never", "again", "without"}
STOP_WORDS = frozenset((ENGLISH_STOP_WORDS | GENERIC_WORDS) - MEANINGFUL)


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'")
    for short, full in CONTRACTIONS.items():
        text = text.replace(short, full)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def stem(word: str) -> str:
    """Tiny suffix stripper - enough to join delivery/delivered/deliver and
    country/countries without pulling in NLTK."""
    if len(word) <= 4 or word.endswith(("ss", "us", "is")):
        return word
    for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("ery", "er"), ("ly", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)] + repl
            # "shipping" -> "shipp" -> "ship"
            if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiouls":
                word = word[:-1]
            return word
    if word.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


# Customers and the FAQ often use different words for the same thing. Both
# sides are mapped to one canonical (stemmed) word, so "delivery charge"
# meets "shipping cost" and "shipment" meets "package". Kept deliberately
# small and domain-specific: a broad thesaurus would blur distinct FAQs.
SYNONYMS = {
    "ship": "deliver", "courier": "deliver",
    "shipment": "package", "parcel": "package", "consignment": "package",
    "abroad": "international", "overseas": "international", "foreign": "international",
    "charge": "cost", "fee": "cost",
    "crack": "broken", "smash": "broken", "shatter": "broken",
    "bill": "invoice", "receipt": "invoice",
    "swap": "exchange", "replace": "exchange",
    "cellphone": "phone", "mobile": "phone",
}


def content_words(text: str) -> list[str]:
    words = (stem(w) for w in normalize(text).split() if w not in STOP_WORDS and len(w) > 1)
    return [SYNONYMS.get(w, w) for w in words]


@dataclass
class Match:
    faq: dict
    score: float

    @property
    def id(self) -> str:
        return self.faq["id"]


def load_faqs(path: Path = config.FAQ_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class FAQMatcher:
    def __init__(self, faqs: list[dict] | None = None):
        self.faqs = faqs if faqs is not None else load_faqs()

        # Every phrasing (canonical question + variants) is indexed separately
        # and an FAQ scores as its best phrasing, so adding variants can only
        # help recall, never dilute it.
        self._phrasings: list[str] = []
        self._owner: list[int] = []
        self._faq_vocab: list[set[str]] = []
        for i, faq in enumerate(self.faqs):
            vocab = set()
            for t in [faq["question"], *faq.get("variants", [])]:
                words = content_words(t)
                self._phrasings.append(" ".join(words))
                self._owner.append(i)
                vocab.update(words)
            self._faq_vocab.append(vocab)
        self._vocab = sorted(set().union(*self._faq_vocab))
        self._known = set(self._vocab)

        self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"\S+")
        self._matrix = self._vec.fit_transform(self._phrasings)

    def correct(self, words: list[str]) -> list[str]:
        """Snap unknown words to the FAQ vocabulary. Short words are left
        alone (a 3-letter edit is too ambiguous to guess), and a correction
        must keep the first letter: real typos rarely change it ("ordr" ->
        order), while unrelated words often differ only there ("rice" is not
        a misspelled "price")."""
        out = []
        for w in words:
            if w not in self._known and len(w) > 3:
                candidates = [v for v in self._vocab if v[0] == w[0]]
                w = _closest(w, candidates) or w
            out.append(w)
        return out

    def match(self, query: str, top_k: int = 3) -> list[Match]:
        words = self.correct(content_words(query))
        if not words:
            return []
        q = " ".join(words)
        sims = cosine_similarity(self._vec.transform([q]), self._matrix)[0]

        best: dict[int, float] = {}
        for p, owner in enumerate(self._owner):
            score = (config.TFIDF_WEIGHT * sims[p]
                     + config.FUZZY_WEIGHT * _sorted_ratio(q, self._phrasings[p]) / 100)
            if score > best.get(owner, 0.0):
                best[owner] = score

        results = []
        for owner, score in best.items():
            coverage = sum(w in self._faq_vocab[owner] for w in words) / len(words)
            if coverage == 0:
                continue  # content-word gate: no topical overlap at all
            # Partial coverage is penalised so "cook pasta for my order"
            # doesn't ride on the single shared word "order".
            results.append(Match(self.faqs[owner], float(score * (0.5 + 0.5 * coverage))))

        results.sort(key=lambda m: m.score, reverse=True)
        return results[:top_k]
