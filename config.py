"""All tunables in one place, so thresholds and model names can be adjusted
for the demo without touching engine code."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FAQ_PATH = DATA_DIR / "faq_data.json"
ORDERS_PATH = DATA_DIR / "orders.json"

# Load API keys from .env if python-dotenv is installed; plain environment
# variables work too, so the dependency is optional.
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

DEVICE = "cpu"  # no GPU path exists anywhere in this project

BOT_NAME = "Aria"
COMPANY_NAME = "ShopEase"

# --- Rule engine: confidence routing --------------------------------------
ANSWER_THRESHOLD = 0.55   # >= this: answer directly
CLARIFY_THRESHOLD = 0.35  # between the two: "did you mean ...?"; below: AI or human

# Blend weights for the matcher: TF-IDF carries meaning, whole-string fuzzy
# rewards reordered phrasing. Typos are handled earlier by spell correction.
TFIDF_WEIGHT = 0.75
FUZZY_WEIGHT = 0.25

# An unknown query word is corrected to the closest FAQ vocabulary word if
# they're at least this similar (0-100): "pasword" -> "password" passes,
# "weather" -> "whether" style near-misses on unrelated words mostly don't.
SPELL_MATCH = 80

# Telugu / Hindi matcher: character n-grams carry more weight than whole words,
# because Telugu attaches case endings to words (ఆర్డర్‌ను, ఆర్డర్‌కి).
INDIC_WORD_WEIGHT = 0.4
INDIC_CHAR_WEIGHT = 0.6
INDIC_WORD_MATCH = 75      # fuzzy similarity (0-100) for a word to count as shared

# Language: "auto" detects each message (English / Telugu / Hindi, including
# romanized Telugu/Hindi). Voice mode needs a fixed language for recognition.
DEFAULT_LANGUAGE = os.getenv("BOT_LANGUAGE", "en")

# --- Sentiment ---------------------------------------------------------------
# VADER compound score bands (-1..1). "angry" additionally needs an anger
# word or shouting, so a plain complaint ("it arrived broken") stays "negative".
ANGRY_SCORE = -0.2
NEGATIVE_SCORE = -0.35
POSITIVE_SCORE = 0.3
# Frustration builds up across the conversation (angry +2, negative +1,
# calm turns -1). At this level the customer goes to a human with priority,
# rather than getting yet another bot answer.
FRUSTRATION_ESCALATE = 3

# --- AI fallback (cloud LLMs, optional) ------------------------------------
# Tried in this order; providers that aren't set up (Ollama not running, no
# API key) are skipped silently. The local model goes first: free and private.
AI_PROVIDER_ORDER = ["ollama", "groq", "gemini", "claude"]

# Local model via Ollama (optional): `ollama pull qwen2.5:1.5b` (~1 GB, runs on
# a CPU; ~1.1 GB RAM). Sized for a 4 GB Raspberry Pi 5; qwen2.5:0.5b is faster,
# qwen2.5:3b smarter if there's RAM to spare.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_MAX_FAQS = 6         # prompt reading dominates CPU latency, so send only the top matches
OLLAMA_NUM_CTX = 2048       # context window; smaller means less RAM
OLLAMA_TIMEOUT = 90         # seconds; a Pi is slow but shouldn't hang forever
OLLAMA_KEEP_ALIVE = "30m"   # keep the model loaded between questions
OLLAMA_LANGUAGES = ("English", "Hindi")  # a 1.5B model's Telugu is too weak to trust

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5")

AI_TIMEOUT_SECONDS = 20
AI_MAX_TOKENS = 1024       # replies are spoken aloud, so the prompt asks for 1-3 sentences
AI_HISTORY_TURNS = 4       # recent exchanges sent for conversational context
AI_ESCALATE_TOKEN = "ESCALATE"

# --- Voice ------------------------------------------------------------------
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "en-IN")  # Google recognizer locale
SAMPLE_RATE = 16000
SILENCE_RMS = 150          # int16 RMS floor; the live threshold never goes below this

# End-of-speech detection (all in seconds unless noted)
VAD_FRAME_MS = 30          # analysis frame length, milliseconds
VAD_WARMUP = 0.4           # discard the mic's first moments (often silent zeros)
VAD_CALIBRATE = 0.5        # then measure room noise before listening
VAD_NOISE_FACTOR = 2.5     # speech must be this many times louder than room noise (75th pct)
VAD_MIN_SPEECH = 0.15      # sustained loudness needed to count as speech starting
VAD_END_SILENCE = 0.9      # this much quiet after speech ends the utterance
VAD_START_TIMEOUT = 10.0   # give up if nobody starts speaking (people pause before talking)
VAD_MAX_SECONDS = 10.0     # hard cap on one utterance
VAD_PREROLL = 0.3          # audio kept from just before speech was detected

TTS_ENGINE = os.getenv("TTS_ENGINE", "edge")   # "edge" (neural, online) or "sapi" (offline)
# Neural voice: Indian English female. Other gentle options:
# en-US-AriaNeural, en-US-JennyNeural, en-GB-SoniaNeural.
TTS_VOICE = os.getenv("TTS_VOICE", "en-IN-NeerjaNeural")
TTS_NEURAL_RATE = "+5%"    # speaking speed: "+0%" is normal, "+10%" faster, "-10%" slower
TTS_NEURAL_PITCH = "+0Hz"

# --- Offline mode -------------------------------------------------------------
# "auto": online services when the internet is up, offline models otherwise.
# "always": never touch the network (also: python main.py --offline).
# "never": online only, as before.
OFFLINE_MODE = os.getenv("OFFLINE_MODE", "auto")
MODELS_DIR = BASE_DIR / "models"   # filled by: python -m tools.download_models

NET_CHECK_HOST = "www.google.com"  # reachable if both Google ASR and Edge TTS are
NET_CHECK_TIMEOUT = 1.5            # seconds; a dead network must not stall a reply
NET_CHECK_TTL = 30                 # seconds to trust the last check

# Offline speech recognition (Vosk, ~40 MB each). Telugu's small model is
# weak on free speech (~88% word error rate), so for Telugu the recognizer is
# limited to words that occur in the FAQ, which is far easier to get right.
VOSK_MODELS = {
    "en": "vosk-model-small-en-in-0.4",
    "hi": "vosk-model-small-hi-0.22",
    "te": "vosk-model-small-te-0.42",
}
VOSK_GRAMMAR_LANGS = ("te",)

# Offline neural voices (Piper, ~63 MB each, ONNX on CPU).
PIPER_VOICES = {
    "en": "en_US-hfc_female-medium",
    "hi": "hi_IN-priyamvada-medium",
    "te": "te_IN-maya-medium",
}
PIPER_LENGTH_SCALE = 1.0   # speaking pace: below 1.0 is faster, above is slower

# Last-resort offline English voice (Windows SAPI5 / Linux espeak-ng via pyttsx3)
TTS_OFFLINE_VOICE = "Zira"  # Windows' built-in female voice
TTS_RATE = 160              # words per minute; default ~200 sounds rushed
TTS_VOLUME = 0.9
