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
            below 0.35    ──> AI fallback (Groq -> Gemini -> Claude)
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
  isn't there, so it can't make up policies.
- **Session stats** (turns, answered, AI-answered, clarified, escalated,
  average latency) print when the session ends.

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
python main.py --mode voice         # press Enter, speak for 5 s, hear the reply
python main.py --speak              # type questions, hear replies
python main.py --no-ai              # rule engine only, fully offline
pytest tests/ -v                    # unit tests (no network needed)
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
| `exit` | session stats |

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
dialogue/manager.py    confidence routing, context, clarification, stats
ai/llm.py              Groq / Gemini / Claude fallback chain
asr/speech_to_text.py  microphone capture + Google speech recognition
tts/text_to_speech.py  neural voice via edge-tts, offline fallback via pyttsx3
main.py                CLI
tests/test_matcher.py  unit tests
```
