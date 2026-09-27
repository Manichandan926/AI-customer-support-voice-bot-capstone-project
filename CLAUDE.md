# AI Customer Support Voice Bot - Project Context

## Overview

Capstone project: **AI Customer Support Voice Bot** (Project 15, Project No. 90).
A conversational voice-AI system that handles customer support queries end to
end - speech or text in, an answer out - instead of a traditional IVR menu.

- **Team:** Namana Mani Chandan Satya Venkata Sai (2300039055), Ilam Akhil
  (2300033560), Seerapu Manikanta (2300039150), Mulaveesala Venkata Siva Naga
  Sai Siddiq (2300031932)
- **Guide:** Dr. B. Samatha
- **Stage:** Review 2 software build. Hardware deployment is a later phase.

## Scope decision (Review 2)

Deliberately simple: **no local ML models**. The earlier plan (local RAG with
fastembed + llama-cpp-python + Qwen GGUF, faster-whisper) was dropped in favour of:

1. **Smart rule engine** (`nlu/`, `dialogue/`) - stemming + spell correction
   against the FAQ vocabulary, then word TF-IDF + fuzzy match over
   `data/faq_data.json`, content-word gate,
   context-aware follow-ups, order-number lookup, small talk. Offline, ms-fast.
2. **Cloud AI fallback** (`ai/llm.py`) - only when matcher confidence is below
   `CLARIFY_THRESHOLD` (or the user rejects a clarification). Providers tried in
   order Groq -> Gemini -> Claude, skipping any without a key in `.env`. The LLM is
   grounded on the full FAQ and must reply `ESCALATE` if it isn't covered.
3. **Voice** - SpeechRecognition's free Google recognizer (audio captured via
   sounddevice, not PyAudio). TTS is edge-tts neural voice `en-IN-NeerjaNeural`
   (online, played via Windows MCI), falling back to pyttsx3 + Zira offline.
   Lazy-loaded; text mode never imports them. Energy-based endpointing
   (`asr.Endpointer`, pure logic, unit-tested) stops recording on silence.
4. **Sentiment** (`nlu/sentiment.py`) - VADER + anger lexicon; empathy prefix
   on negative turns, priority escalation when frustration accumulates.
5. **Multilingual** (`nlu/language.py`, `nlu/indic_matcher.py`, `data/i18n/`) -
   English/Telugu/Hindi detected per message; all user-facing text lives in the
   language packs, never in code. Telugu/Hindi matching is word + char n-gram
   TF-IDF with spelling folding. Voice mode takes `--lang` for ASR.
6. **Offline mode** (`network.py`, `asr/offline_asr.py`, `tts/offline_tts.py`) -
   `OFFLINE_MODE` auto/always/never. ASR: Google -> Vosk; TTS: edge-tts -> Piper
   -> pyttsx3 (English only). Models in `models/` (git-ignored), fetched by
   `python -m tools.download_models`. Piper's `use_cuda` stays off.
7. **Local AI** (`ai/llm.py` OllamaProvider) - first in AI_PROVIDER_ORDER,
   auto-detected via Ollama's localhost API (stdlib urllib, no SDK); gets only
   the top OLLAMA_MAX_FAQS ranked FAQs; skipped for Telugu and when nothing
   matched. `--offline` keeps local AI, drops cloud.
8. **Evaluation** (`tools/evaluate.py`, `data/eval/test_set.json`) - held-out
   phrasings; never copy test questions into FAQ data (that would fake the
   score). `tests/test_evaluation.py` guards the measured accuracy floor.

Performance: no scikit-learn - `nlu/tfidf.py` reimplements TfidfVectorizer
(sublinear tf, smooth idf, L2) and was verified identical to 1e-15. Core text
mode: ~0.1 s startup, ~31 MB, ~1 ms replies. Keep heavy imports lazy. Cloud
SDKs live in requirements-cloud.txt.

Tests: `tests/conftest.py` simulates connectivity via `network._connect`;
never patch `socket.create_connection` globally (it breaks urllib).

Must run on Windows and Linux, including a Raspberry Pi 5 (4GB) - keep
everything light and cross-platform (Linux playback via mpg123 etc.).

## Target environment - CPU only

Windows 11, PowerShell, Intel i3 10th gen, 8GB RAM, Python 3.14. No CUDA, no
torch, no GPU-only deps. Before adding a library, check it doesn't pull in
PyTorch.

## Commands

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py [--mode text|voice] [--speak] [--no-ai]
pytest tests/
```

## Conventions

- Comments explain *why*, not *what*.
- Thresholds and model names live in `config.py`, never inline.
- Anything heavy or optional (voice libs, provider SDKs) is imported lazily
  inside a method and fails with an actionable message ("pip install X").
- Routing/AI behaviour is tested with fake providers - tests never hit the network.
- `DialogueManager.handle(query) -> Turn` and `.stats()` are the engine interface
  `main.py` depends on.

## Out of scope for Review 2

Hardware integration, GPU, multilingual, sentiment, summarization, GUI.
