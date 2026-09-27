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

# --- AI fallback (cloud LLMs, optional) ------------------------------------
# Tried in this order; providers without an API key are skipped silently.
AI_PROVIDER_ORDER = ["groq", "gemini", "claude"]

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5")

AI_TIMEOUT_SECONDS = 20
AI_MAX_TOKENS = 1024       # replies are spoken aloud, so the prompt asks for 1-3 sentences
AI_CONTEXT_FAQS = 4        # top-k FAQ entries handed to the LLM as grounding
AI_HISTORY_TURNS = 4       # recent exchanges sent for conversational context
AI_ESCALATE_TOKEN = "ESCALATE"

# --- Voice ------------------------------------------------------------------
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "en-IN")  # Google recognizer locale
RECORD_SECONDS = 5
SAMPLE_RATE = 16000
SILENCE_RMS = 150          # int16 RMS below this is treated as "nothing said"

TTS_ENGINE = os.getenv("TTS_ENGINE", "edge")   # "edge" (neural, online) or "sapi" (offline)
# Neural voice: Indian English female. Other gentle options:
# en-US-AriaNeural, en-US-JennyNeural, en-GB-SoniaNeural.
TTS_VOICE = os.getenv("TTS_VOICE", "en-IN-NeerjaNeural")
TTS_NEURAL_RATE = "+5%"    # speaking speed: "+0%" is normal, "+10%" faster, "-10%" slower
TTS_NEURAL_PITCH = "+0Hz"

# Offline fallback (Windows SAPI5 via pyttsx3)
TTS_OFFLINE_VOICE = "Zira"  # Windows' built-in female voice
TTS_RATE = 160              # words per minute; default ~200 sounds rushed
TTS_VOLUME = 0.9
