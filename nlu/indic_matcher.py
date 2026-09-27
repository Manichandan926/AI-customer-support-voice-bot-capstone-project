"""FAQ retrieval for Telugu and Hindi.

The English matcher's stemmer and spell-corrector are English-specific, so
Indian languages use a script-agnostic approach instead: word TF-IDF plus
character n-gram TF-IDF. The character n-grams matter most for Telugu,
which glues case endings onto words (ఆర్డర్ -> ఆర్డర్‌ను, ఆర్డర్‌కి), so whole-word
matching alone misses obvious matches. Returns the same Match objects as the
English matcher, pointing at the English FAQ entry (id, category), so the
dialogue manager treats every language the same way.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from nlu.language import LanguagePack, normalize_any
from nlu.matcher import Match

try:
    from rapidfuzz import fuzz

    def _ratio(a: str, b: str) -> float:
        return fuzz.ratio(a, b)
except ImportError:
    from difflib import SequenceMatcher

    def _ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio() * 100


class IndicMatcher:
    def __init__(self, pack: LanguagePack, faqs: list[dict]):
        self.pack = pack
        self._by_id = {f["id"]: f for f in faqs}
        self._stop = pack.words.get("stopwords", set())

        self._phrasings: list[str] = []
        self._owner: list[str] = []
        self._vocab: dict[str, set[str]] = {}
        for faq_id, entry in pack.faq.items():
            vocab = set()
            for text in [entry["question"], *entry.get("variants", [])]:
                words = self.content_words(text)
                self._phrasings.append(" ".join(words))
                self._owner.append(faq_id)
                vocab.update(words)
            self._vocab[faq_id] = vocab

        self._word_vec = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), sublinear_tf=True)
        self._char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self._word_m = self._word_vec.fit_transform(self._phrasings)
        self._char_m = self._char_vec.fit_transform(self._phrasings)

    def content_words(self, text: str) -> list[str]:
        return [w for w in normalize_any(text).split() if w not in self._stop]

    def match(self, query: str, top_k: int = 3) -> list[Match]:
        words = self.content_words(query)
        if not words:
            return []
        q = " ".join(words)
        word_sims = cosine_similarity(self._word_vec.transform([q]), self._word_m)[0]
        char_sims = cosine_similarity(self._char_vec.transform([q]), self._char_m)[0]

        best: dict[str, float] = {}
        for p, owner in enumerate(self._owner):
            score = config.INDIC_WORD_WEIGHT * word_sims[p] + config.INDIC_CHAR_WEIGHT * char_sims[p]
            if score > best.get(owner, 0.0):
                best[owner] = score

        results = []
        for owner, score in best.items():
            coverage = self._coverage(words, self._vocab[owner])
            if coverage == 0:
                continue  # same content-word gate as English: no topical overlap, no match
            results.append(Match(self._by_id[owner], float(score * (0.5 + 0.5 * coverage))))
        results.sort(key=lambda m: m.score, reverse=True)
        return results[:top_k]

    @staticmethod
    def _coverage(words: list[str], vocab: set[str]) -> float:
        hits = 0
        for w in words:
            # Fuzzy, because inflected forms (ఆర్డర్‌ను vs ఆర్డర్) differ by a suffix.
            if w in vocab or any(_ratio(w, v) >= config.INDIC_WORD_MATCH for v in vocab):
                hits += 1
        return hits / len(words)
