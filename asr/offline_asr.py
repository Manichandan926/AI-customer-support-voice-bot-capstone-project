"""Offline speech recognition with Vosk (Kaldi) small models, ~40 MB each,
fast enough on a Raspberry Pi. Used when the internet is down or with
--offline.
"""

import json

import config

_models = {}  # loaded once per language; a Vosk model takes a second or two to load


def model_path(lang: str):
    name = config.VOSK_MODELS.get(lang)
    return config.MODELS_DIR / "vosk" / name if name else None


def is_available(lang: str) -> bool:
    path = model_path(lang)
    return path is not None and (path / "conf").is_dir()


def grammar_for(lang: str) -> list[str]:
    """Words and phrases the customer might say in this language, taken from
    the language pack: FAQ questions and variants plus yes/no/exit and the
    other word lists. Limiting recognition to this vocabulary turns a model
    that gets most free-speech words wrong into a usable one for this
    narrow domain. "[unk]" absorbs anything else."""
    from nlu.language import load_pack
    pack = load_pack(lang)
    phrases, words = set(), set()
    texts = [t for entry in pack.faq.values() for t in (entry["question"], *entry["variants"])]
    texts += [w for key, ws in pack.data.get("words", {}).items() if key != "stopwords" for w in ws]
    for text in texts:
        # The model only outputs native script, so Latin tokens - romanized
        # variants, and acronyms like UPI/GST inside Telugu phrases - are
        # unrecognizable to it and are dropped.
        tokens = [t for t in text.replace("?", " ").replace(",", " ").split() if not t.isascii()]
        if not tokens:
            continue
        phrases.add(" ".join(tokens))
        words.update(tokens)
    return sorted(phrases | words) + ["[unk]"]


class VoskRecognizer:
    def __init__(self, lang: str):
        self.lang = lang

    def _model(self):
        if self.lang not in _models:
            try:
                from vosk import Model, SetLogLevel
            except ImportError:
                raise RuntimeError("vosk isn't installed, run: pip install vosk") from None
            if not is_available(self.lang):
                raise RuntimeError(f"no offline speech model for '{self.lang}', run: "
                                   f"python -m tools.download_models --lang {self.lang}")
            SetLogLevel(-1)  # Kaldi is chatty on stderr
            _models[self.lang] = Model(str(model_path(self.lang)))
        return _models[self.lang]

    def transcribe(self, samples, sample_rate: int = config.SAMPLE_RATE) -> str:
        from vosk import KaldiRecognizer
        model = self._model()
        if self.lang in config.VOSK_GRAMMAR_LANGS:
            rec = KaldiRecognizer(model, sample_rate, json.dumps(grammar_for(self.lang), ensure_ascii=False))
        else:
            rec = KaldiRecognizer(model, sample_rate)
        rec.AcceptWaveform(samples.tobytes())
        text = json.loads(rec.FinalResult()).get("text", "")
        return " ".join(w for w in text.split() if w != "[unk]")
