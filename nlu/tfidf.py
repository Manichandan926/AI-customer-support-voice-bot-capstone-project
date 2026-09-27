"""Minimal TF-IDF + cosine similarity, replacing scikit-learn.

Same math as sklearn's TfidfVectorizer(sublinear_tf=True) with its defaults
(smooth idf, L2 norm), so scores - and therefore every threshold tuned on
them - are unchanged. The FAQ index is a few hundred short phrasings, so
plain dicts are fast enough, and dropping scikit-learn (with scipy and
joblib) takes ~1 s off startup and a large install off the Raspberry Pi.
"""

import math
from collections import Counter, defaultdict


def word_ngrams(text: str, max_n: int = 2) -> list[str]:
    tokens = text.lower().split()
    grams = list(tokens)
    for n in range(2, max_n + 1):
        grams += [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return grams


def char_wb_ngrams(text: str, min_n: int, max_n: int) -> list[str]:
    """Character n-grams inside word boundaries, padded with a space on each
    side - identical to sklearn's analyzer="char_wb", including its rule that
    a word shorter than n contributes itself once."""
    grams = []
    for word in text.lower().split():
        w = f" {word} "
        for n in range(min_n, max_n + 1):
            offset = 0
            grams.append(w[offset:offset + n])
            while offset + n < len(w):
                offset += 1
                grams.append(w[offset:offset + n])
            if offset == 0:
                break
    return grams


class TfidfIndex:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self._idf: dict[str, float] = {}
        self._postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self._n_docs = 0

    def fit(self, docs: list[str]) -> "TfidfIndex":
        counts = [Counter(self.analyzer(d)) for d in docs]
        self._n_docs = len(docs)
        df = Counter(term for c in counts for term in c)
        self._idf = {t: math.log((1 + self._n_docs) / (1 + f)) + 1 for t, f in df.items()}
        for i, c in enumerate(counts):
            for term, weight in self._weights(c).items():
                self._postings[term].append((i, weight))
        return self

    def _weights(self, counts: Counter) -> dict[str, float]:
        w = {t: (1 + math.log(n)) * self._idf[t] for t, n in counts.items() if t in self._idf}
        norm = math.sqrt(sum(v * v for v in w.values()))
        return {t: v / norm for t, v in w.items()} if norm else {}

    def similarities(self, text: str) -> list[float]:
        """Cosine similarity of text against every indexed document."""
        scores = [0.0] * self._n_docs
        for term, qw in self._weights(Counter(self.analyzer(text))).items():
            for doc, dw in self._postings[term]:
                scores[doc] += qw * dw
        return scores
