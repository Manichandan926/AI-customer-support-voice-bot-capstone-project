# AI Customer Support Voice Bot

Capstone Project 15 - a conversational support assistant ("Aria" for the
fictional store *ShopEase*) that handles customer queries by text or voice,
instead of a traditional IVR menu.

Runs on a plain CPU laptop (tested target: Intel i3, 8 GB RAM, Windows 11).
No GPU, no local ML models, no multi-GB downloads.

## How it works

```
 voice ──> Google Speech Recognition ──┐
                                       v
 text ──────────────────────────> Dialogue Manager ──> reply ──> neural voice (Neerja, Indian English)
                                                                  (offline fallback: Windows Zira)
                                       │
       1. exit / yes-no to a pending "did you mean?"
       2. small talk, "talk to a human"
       3. order number?  ──> order lookup ("order 10234 is shipped...")
       4. FAQ matcher (stemming + spell correction + TF-IDF + fuzzy, context-aware)
            score >= 0.55 ──> answer instantly
            0.35 - 0.55   ──> "Just to confirm, are you asking ...?"
            below 0.35    ──> AI fallback (local Ollama -> Groq -> Gemini -> Claude)
                                 grounded on the FAQ knowledge base
                                 no answer / no keys ──> escalate to human
```

- **Smart rule engine** answers most questions in about 1-5 ms, fully offline:
  tolerant of typos ("pasword", "refnd"), contractions and rephrasing, with a
  content-word gate so off-topic questions ("what's the weather?") are never
  mistaken for FAQ matches.
- **Context memory**: follow-ups like "how long does that take?" after a
  returns question resolve to the refund timeline.
- **AI fallback** handles questions the FAQ list doesn't phrase directly. The
  LLM sees only the FAQ content and must reply `ESCALATE` if the answer
  isn't there, so it can't make up policies. A **local model** (Ollama,
  qwen2.5:1.5b, CPU-only, ~1 GB) is used first when installed - no API key,
  works offline, fits a 4 GB Raspberry Pi - then any cloud API keys.
- **English, Telugu and Hindi**: the language is detected per message
  (script, or common romanized words like "naa order ekkada undi" / "mera
  order kahan hai") and the whole reply - FAQ answers, order status,
  clarifications, escalations - comes back in that language, spoken with a
  matching neural voice (Neerja, Shruti, Swara).
- **Sentiment detection** (offline, VADER + an anger lexicon): upset customers
  get an empathetic reply; repeated anger hands them to a senior agent with
  priority instead of another bot answer.
- **Works offline too**: with no internet, speech recognition switches to
  Vosk and replies to Piper neural voices, automatically - both run on the CPU
  and fit on a Raspberry Pi. Online, the better Google/Microsoft services are
  used.
- **Hands-free listening**: voice mode stops recording when you stop talking,
  adapting to the room's background noise.
- **Session stats** (turns, answered, AI-answered, clarified, escalated,
  upset turns, average latency) print when the session ends.

## Setup

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then paste in any API keys you have (all optional)
```

API keys (any subset works, none is required):

| Provider | Cost | Get a key |
|---|---|---|
| Groq (Llama 3.3 70B) | free tier | https://console.groq.com/keys |
| Google Gemini | free tier | https://aistudio.google.com/apikey |
| Anthropic Claude | paid per use | https://console.anthropic.com/ |

## Run

```powershell
python main.py                      # text chat
python main.py --mode voice         # press Enter, speak; it stops when you stop
python main.py --speak              # type questions, hear replies
python main.py --no-ai              # rule engine only, fully offline
python main.py --mode voice --lang te   # listen and reply in Telugu (hi = Hindi)
python main.py --mode voice --offline   # force offline speech + voices (no internet used)
pytest tests/ -v                    # unit tests (no network, mic or speaker needed)
python -m asr.speech_to_text        # mic check: shows noise level, threshold, transcript
python -m tools.evaluate            # accuracy on 105 held-out questions (see docs/evaluation.md)
python main.py --hands-free         # voice without pressing Enter (standalone device)
```

Type or say `exit` to end the session and see the stats.

### Demo script

| You say | What it shows |
|---|---|
| `hi` | small talk |
| `how do I reset my password` | confident FAQ answer |
| `i forgot my pasword` | typo tolerance |
| `where is my order 10567` | order-number extraction + lookup |
| `how do I return an item` then `how long does that take` | context memory |
| `my package was left open` then `yes` | clarify-then-confirm |
| `track order` then `10234` | asks for, then looks up, the order number |
| `do you have a store in hyderabad` | AI fallback (with a key) |
| `what's the weather today` | off-topic, escalated to human |
| `the product arrived broken` then `this is the worst service, useless` | empathy, then priority handoff on repeated anger |
| `నా ఆర్డర్ ఎక్కడ ఉంది` then `10234` | Telugu FAQ answer + order status in Telugu |
| `mera refund kab aayega` | romanized Hindi, answered in Hindi |
| `exit` | session stats (including languages used) |

Sample order numbers: 10234, 10567, 10891, 11002, 11345, 11789, 12001.

## Troubleshooting

- **No audio / COM error from pyttsx3**: pywin32's post-install step sometimes
  doesn't run under modern pip. Fix with:
  `python venv\Scripts\pywin32_postinstall.py -install`
- **"Couldn't record from the microphone"**: check Windows Settings > Privacy &
  security > Microphone, and that desktop apps are allowed to use it.
- **"Google speech service unreachable"**: voice recognition needs internet.
  You can still type in voice mode - the reply will be spoken.
- **AI fallback shows "off"**: no keys found. Check `.env` is in the project
  folder and the variable names match `.env.example`.
- **Wrong accent recognition**: set `ASR_LANGUAGE` in `.env` (default `en-IN`).
- **Voice mode never hears you, or never stops listening**: run
  `python -m asr.speech_to_text` and compare the printed noise level with your
  speech level. In a noisy room raise `VAD_NOISE_FACTOR` in `config.py`; if it
  misses quiet speech, lower it.

## Linux / Raspberry Pi

One command on a Raspberry Pi 5 (Raspberry Pi OS 64-bit) or any Debian/Ubuntu:

```bash
bash scripts/setup_pi.sh              # packages, venv, requirements, self-test
bash scripts/setup_pi.sh --all te     # + offline models, local AI, start hands-free in Telugu at boot
```

It installs the audio packages, creates the venv, runs the tests and the
evaluation, and optionally downloads offline models (`--models`), installs
Ollama with a model sized to the RAM (`--llm`), and registers a systemd
service that starts Aria hands-free at boot (`--service en|te|hi`; each
"bye" ends a customer session and the next one starts fresh).

Memory budget on a 4 GB Pi 5: bot ~150 MB, offline models ~250 MB, local
model ~1.1 GB - under 2 GB in total. Manual install, if preferred:

```bash
sudo apt install -y python3-venv libportaudio2 mpg123 espeak-ng
python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
python main.py --mode voice
```

`mpg123` plays the neural voice, `libportaudio2` gives Python mic access, and
`espeak-ng` is the last-resort offline voice. For proper offline voices and
speech recognition, also run `python -m tools.download_models`.

## Measured accuracy

`python -m tools.evaluate` runs 105 **held-out** questions - phrased
differently from anything in the FAQ data, so it measures generalization,
not memorization - through the rule engine alone (no AI, offline):

| | Overall | English | Telugu | Hindi |
|---|---|---|---|---|
| resolved (answered, or confirmed via "did you mean?") | 88% | 82% | 100% | 100% |
| answer precision (right when it answers) | 89% | 83% | 100% | 100% |
| off-topic questions correctly handed off | 92% | 88% | 100% | 100% |
| language detected correctly | 100% | 100% | 100% | 100% |
| average reply time | 5 ms | 1.4 ms | 13 ms | 11 ms |

The remaining English misses need understanding of meaning ("do I have to
*pay for* delivery" vs "shipping *cost*"), which is what the AI fallback is
for; run `python -m tools.evaluate --with-ai` with Ollama installed to
measure it. A test (`tests/test_evaluation.py`) fails if accuracy ever
drops below these levels. Full list of misses: `docs/evaluation.md`.

## Local AI (optional, no API key)

```bash
# install Ollama from https://ollama.com, then:
ollama pull qwen2.5:1.5b      # ~1 GB download, ~1.1 GB RAM while running
python main.py                # the banner shows "AI fallback: ollama (qwen2.5:1.5b)"
```

Detected automatically - without Ollama the bot behaves exactly as before.
Runs on the CPU (a few seconds per answer on a laptop, longer on a Pi), so
it only sees the 6 most relevant FAQs rather than all 40: reading the
prompt is the slow part on a CPU. It answers English and Hindi; Telugu is
beyond a model this size, so Telugu questions go to cloud AI or a human.

## Offline mode

One-time download of the offline models (~330 MB, into `models/`, which git ignores):

```bash
python -m tools.download_models            # English, Hindi, Telugu: speech models + voices
python -m tools.download_models --status   # what's installed
```

| | Online (default when internet is up) | Offline (automatic, or `--offline`) |
|---|---|---|
| Speech recognition | Google (free, no key) | Vosk small models - English, Hindi; Telugu limited to FAQ vocabulary |
| Voice | Microsoft neural: Neerja, Shruti, Swara | Piper neural: hfc_female, Priyamvada, Maya |
| Without any download | - | English: Windows/espeak system voice; typing works in every language |

The bot checks connectivity once (cached for 30 s) and falls back per
utterance, so a dropped connection mid-demo costs one short timeout, not a
frozen bot. Offline Telugu *recognition* is the weak spot: the only small
Telugu model mishears most free speech, so it's restricted to words that
appear in the Telugu FAQ - fine for FAQ-style questions, not for open chat.

## Telugu and Hindi

All translated text lives in one file per language, `data/i18n/te.json` and
`data/i18n/hi.json` - FAQ questions and answers, bot messages, order status
wording, and word lists (yes/no/exit, anger words, filler words). A native
speaker can improve wording there without touching code; the tests check
every FAQ, message and order stays covered.

- **Text mode** switches language automatically on every message.
- **Voice mode** needs the language up front for speech recognition:
  `--lang te` or `--lang hi`. Recognition and the Telugu/Hindi voices need
  internet; offline, Telugu/Hindi replies are shown as text.
- Matching ignores spelling variants that people and speech recognizers mix
  up (రీఫండ్/రిఫండ్, कहाँ/कहां/कहा, ऑर्डर/आर्डर).
- On Windows, use Windows Terminal (default on Windows 11) so Telugu script
  renders; the old console may show boxes.

## Aria's voice

Replies are spoken with Microsoft's Indian English neural voice **en-IN-NeerjaNeural** (via
`edge-tts`: free, no key, about 1 s to generate a reply, needs internet). If
it's unreachable, the bot switches to Windows' offline female voice (Zira)
automatically. To change it, set these in `.env`:

```
TTS_VOICE=en-US-AriaNeural     # or en-US-JennyNeural, en-GB-SoniaNeural
TTS_ENGINE=sapi                # force the offline voice
```

## Project layout

```
config.py              thresholds, model names, voice settings
data/faq_data.json     40 FAQ entries (orders, shipping, returns, refunds, billing, account)
data/orders.json       mock order table for order-number lookups
nlu/matcher.py         stemming, spell correction, TF-IDF + fuzzy retrieval, content-word gate
nlu/entities.py        order-number extraction and lookup
nlu/sentiment.py       offline sentiment + anger detection
nlu/language.py        language packs, per-message language detection, spelling folding
nlu/indic_matcher.py   Telugu/Hindi FAQ matching (word + character n-gram TF-IDF)
data/i18n/*.json       English / Telugu / Hindi text: FAQs, messages, orders, word lists
dialogue/manager.py    confidence routing, context, clarification, stats
ai/llm.py              Groq / Gemini / Claude fallback chain
asr/speech_to_text.py  mic capture with end-of-speech detection + Google speech recognition
tts/text_to_speech.py  voice chain: edge-tts online -> Piper offline -> system voice
tts/offline_tts.py     Piper offline neural voices
asr/offline_asr.py     Vosk offline speech recognition (+ Telugu FAQ grammar)
network.py             cached connectivity check that drives online/offline switching
tools/download_models.py  one-time offline model download
tools/evaluate.py      accuracy report on data/eval/test_set.json (held-out questions)
scripts/setup_pi.sh    Raspberry Pi / Linux setup, optional autostart service
docs/evaluation.md     latest evaluation report
main.py                CLI
tests/                 unit tests (matcher, dialogue, AI fallback, sentiment, endpointing, playback, multilingual)
```
