"""Language packs and per-message language detection (English, Telugu, Hindi).

Detection is by script first - Telugu and Devanagari have their own Unicode
blocks, so that part is exact - then by common romanized words ("naa order
ekkada undi", "mera order kahan hai") for people typing Indian languages in
English letters. No model, no network.
"""

import json
import re
import unicodedata
from functools import lru_cache

import config

SUPPORTED = ("en", "te", "hi")
_TELUGU = (0x0C00, 0x0C7F)
_DEVANAGARI = (0x0900, 0x097F)


class LanguagePack:
    def __init__(self, data: dict):
        self.data = data
        self.code: str = data["lang"]
        self.name: str = data["name"]
        self.asr_code: str = data["asr_code"]
        self.tts_voice: str = data["tts_voice"]
        self.messages: dict = data["messages"]
        self.faq: dict = data.get("faq", {})
        self.status: dict = data.get("status", {})
        self.orders: dict = data.get("orders", {})
        self.apologetic = tuple(p.lower() for p in data.get("apologetic_prefixes", []))
        # Word lists are compared after the same normalization as user input.
        self.words = {k: {normalize_any(w) for w in v} for k, v in data.get("words", {}).items()}

    def msg(self, key: str, **fmt) -> str:
        text = self.messages[key]
        return text.format(**fmt) if fmt else text

    def is_word(self, category: str, norm_text: str) -> bool:
        return norm_text in self.words.get(category, ())

    def contains_word(self, category: str, norm_text: str) -> bool:
        padded = f" {norm_text} "
        return any(f" {w} " in padded for w in self.words.get(category, ()))


@lru_cache(maxsize=None)
def load_pack(lang: str) -> LanguagePack:
    if lang not in SUPPORTED:
        raise ValueError(f"unsupported language {lang!r}; choose one of {', '.join(SUPPORTED)}")
    with open(config.DATA_DIR / "i18n" / f"{lang}.json", encoding="utf-8") as f:
        return LanguagePack(json.load(f))


# Spelling variants that people and speech recognizers mix up freely, folded
# to one form for matching only (replies keep their proper spelling):
# long/short vowels (రీఫండ్ / రిఫండ్), nasal marks (कहाँ / कहां / कहा), ऑ vs आ
# (ऑर्डर / आर्डर) and the nukta dot (ज़रूर / जरूर).
_FOLD = str.maketrans({
    # Telugu: long -> short vowel signs and letters
    "ీ": "ి", "ూ": "ు", "ే": "ె", "ో": "ొ",
    "ఈ": "ఇ", "ఊ": "ఉ", "ఏ": "ఎ", "ఓ": "ఒ",
    # Devanagari: long -> short vowels, drop nasal marks and the nukta,
    # English "o" sound (ऑर्डर) -> आ, the casual spelling (आर्डर)
    "ी": "ि", "ू": "ु", "ई": "इ", "ऊ": "उ",
    "ँ": None, "ं": None, "़": None, "ऑ": "आ", "ॉ": "ा",
})


def normalize_any(text: str) -> str:
    """Lowercase and strip punctuation for any script. Unlike a [a-z0-9]
    filter, this keeps Telugu/Hindi vowel signs (combining marks), which are
    part of the word, while dropping the danda (।) and other punctuation."""
    text = unicodedata.normalize("NFC", text).lower().replace("’", "'").translate(_FOLD)
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in "PS" or ch in "‌‍":  # punctuation, symbols, zero-width joiners
            out.append(" ")
        else:
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def detect_language(text: str, current: str = "en") -> str:
    telugu = devanagari = latin = 0
    for ch in text:
        cp = ord(ch)
        if _TELUGU[0] <= cp <= _TELUGU[1]:
            telugu += 1
        elif _DEVANAGARI[0] <= cp <= _DEVANAGARI[1]:
            devanagari += 1
        elif ch.isascii() and ch.isalpha():
            latin += 1
    if telugu or devanagari:
        # Code-mixed input ("నా order ఎక్కడ") still belongs to the Indian
        # language as long as a fair share of the letters are in its script.
        if telugu + devanagari >= 0.3 * (telugu + devanagari + latin):
            return "te" if telugu >= devanagari else "hi"
    words = normalize_any(text).split()
    if not latin or not words:
        return current  # digits only ("10234"), emoji, empty: keep the conversation's language
    if len(words) <= 2 and current != "en":
        # "ok", "yes", "10234 please" mid-conversation shouldn't flip language.
        return current
    best, best_hits = "en", 0
    for lang in ("te", "hi"):
        hints = load_pack(lang).words.get("romanized_hints", set())
        hits = sum(w in hints for w in words)
        if hits > best_hits:
            best, best_hits = lang, hits
    if best_hits >= 2 or (best_hits == 1 and best_hits / len(words) >= 0.34):
        return best
    return "en"
