# AI Customer Support Voice Bot — Text Prototype

A basic-level, console-based prototype of the AI Customer Support
Voice Bot's core logic. This simulates the pipeline using keyboard
input instead of live audio, so the team can demonstrate and refine
the decision-making logic (intent classification, response
generation, escalation) before the real ASR/NLU/TTS models are
integrated on the Orange Pi 5 hardware.

## Why it's structured this way

The real system's pipeline is:

```
Speech Input -> Speech-to-Text (ASR) -> Intent Classification ->
Response Generation -> Text-to-Speech -> Spoken Output
```

This prototype implements the same four stages as separate,
swappable modules:

| Stage | Prototype implementation | Real implementation (later) |
|---|---|---|
| Input | `voicebot/input/console_input.py` — keyboard text | ASR (e.g. whisper.cpp) |
| NLU | `voicebot/nlu/intent_classifier.py` — keyword matching | Fine-tuned classifier (e.g. DistilBERT/ONNX) |
| Dialogue | `voicebot/dialogue/response_generator.py` — canned responses + escalation rule | LLM or retrieval-augmented response generation |
| Output | `voicebot/output/console_output.py` — console print | TTS (e.g. piper-tts) |

`voicebot/pipeline.py` wires these four stages together. To upgrade
any one stage later, write a new class with the same interface
(e.g. a class with a `.listen()` method for input, or a `.classify()`
method for NLU) and pass it into `VoiceBotPipeline(...)` — the other
three stages and the orchestration logic do not need to change.

## Project structure

```
voicebot/
├── __init__.py
├── pipeline.py                  # orchestrates all 4 stages
├── input/
│   └── console_input.py         # mocked ASR (keyboard input)
├── nlu/
│   └── intent_classifier.py     # rule-based intent matching
├── dialogue/
│   └── response_generator.py    # canned responses + escalation logic
└── output/
    └── console_output.py        # mocked TTS (console print)
tests/
└── test_pipeline.py             # unit tests for NLU + dialogue stages
```

## Running it

Requires Python 3.8+, no external dependencies.

```bash
python3 -m voicebot.pipeline
```

Example session:

```
AI Customer Support Voice Bot (text prototype)
Type your query below. Type 'exit' to quit.

You: I forgot my password and can't log in
Bot: I can help with account issues. To reset your password or unlock
your account, please visit the 'Account Settings' page, or I can
connect you to an agent for identity verification.

You: asdkjaslkdj random gibberish
Bot: I want to make sure you get the right help here — let me connect
you with a human agent for this one. [ESCALATED]

You: exit
Bot: Thanks for reaching out. Goodbye!
```

## Running the tests

```bash
python3 -m unittest discover -v
```

## Supported intents (current prototype scope)

- `account_issue` — login, password, account access
- `technical_support` — errors, crashes, setup/connection problems
- `billing_inquiry` — invoices, charges, refunds, subscriptions
- `order_status` — shipment/delivery tracking
- `general_inquiry` — fallback for anything unmatched

Queries the classifier is not confident about are escalated to a
human agent rather than answered with a guess, matching the
escalation design in the project's system architecture.

## Next steps (planned upgrades)

- Replace `console_input.py` with a real ASR module.
- Replace `intent_classifier.py` with a trained NLU model.
- Expand `response_generator.py` beyond canned responses (e.g. RAG).
- Replace `console_output.py` with a real TTS module.
