# Demo day checklist

## The night before

- [ ] `git pull` the branch, then `pip install -r requirements.txt`
- [ ] `pytest tests -q` - all green
- [ ] `python -m tools.evaluate` - numbers match the README table
- [ ] `python -m tools.download_models --status` - offline models present (if you'll show offline mode)
- [ ] Charge the laptop; bring **wired earphones with a mic** (a panel room is noisier than home)
- [ ] Windows Terminal font shows Telugu: `python main.py`, type `నమస్కారం`

## Ten minutes before

1. Connect to Wi-Fi (online voices and speech recognition sound best).
2. `python -m asr.speech_to_text --levels` in the room - if speech isn't at
   least ~2x the room noise, use the earphone mic.
3. Volume up; test one spoken reply: `python main.py --speak`, type `hi`.

## The demo (about 5 minutes)

| Step | Command / say | What it shows |
|---|---|---|
| 1 | `python -m tools.demo` (press Enter per scene) | all features in order: FAQ, typos, orders, context, clarify, sentiment, Telugu, Hindi, off-topic, stats |
| 2 | `python main.py --mode voice` then say *"where is my order"*, then *"one zero two three four"* | live voice in, voice out |
| 3 | `python main.py --mode voice --lang te` then say *"నా ఆర్డర్ ఎక్కడ ఉంది"* | Telugu speech in and out |
| 4 | `python -m tools.evaluate` | measured accuracy on 105 unseen questions |
| 5 | `pytest tests -q` | 129 automated tests |
| 6 | (optional) turn Wi-Fi off, `python main.py --mode voice --offline` | works without internet |

## If something goes wrong

| Problem | Fallback |
|---|---|
| Mic doesn't hear you | type in voice mode - the reply is still spoken; or run `python -m tools.demo --speak` |
| No internet | `--offline` (needs models downloaded) or text mode; everything but voices works offline |
| No sound | show text mode; `python -m tools.demo` needs no audio at all |
| Telugu shows boxes | use Windows Terminal (default on Windows 11), not the old console |

## Likely questions

- **"Is this real AI or just string matching?"** - Layered: NLP retrieval
  (TF-IDF, stemming, spell correction, char n-grams for Telugu) answers ~88%
  of unseen questions in ~1 ms with no model; an LLM (local via Ollama, or
  cloud) handles the rest; sentiment analysis routes upset customers. The
  evaluation shows the rules alone can't understand meaning ("pay for
  delivery" vs "shipping cost") - that's the LLM's job, by design.
- **"How do you know it works?"** - 105 held-out questions phrased unlike
  the training data; answer precision 89%, zero wrong answers in
  Telugu/Hindi; a test fails if accuracy regresses.
- **"Why not a big model for everything?"** - It must run on a 4 GB
  Raspberry Pi: startup 0.1 s, 31 MB RAM for the core, replies in ~1 ms;
  wrong answers are worse than "let me connect you" in customer support.
- **"Can it hallucinate?"** - The LLM only sees FAQ text and must reply
  ESCALATE when the answer isn't there; order data comes from the order
  table, never from the model.
- **Limitations (say them before they're asked):** offline Telugu speech
  recognition is weak (small model, restricted to FAQ vocabulary);
  translations need native-speaker review; order data is mock.
