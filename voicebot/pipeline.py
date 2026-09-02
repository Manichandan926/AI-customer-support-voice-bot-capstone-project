"""
pipeline.py

Wires the four pipeline stages together:

    input (ASR)  ->  nlu (intent)  ->  dialogue (response)  ->  output (TTS)

Each stage is a small, swappable object. To go from this text-only
prototype to the real voice pipeline, you only ever replace the
class passed in here for one stage — the VoiceBotPipeline class
itself does not change.
"""

import os

from voicebot.input.console_input import ConsoleInputSource
from voicebot.nlu.intent_classifier import KeywordIntentClassifier
from voicebot.output.console_output import ConsoleOutputSink


def _default_response_generator():
    """Pick the best available response generator at startup.

    • If a GROQ_API_KEY is set and the optional dependency is installed,
      prefer the Groq LLM generator.
    • Otherwise → fall back to the rule-based canned responses.
    """
    if os.environ.get("GROQ_API_KEY"):
        try:
            from voicebot.dialogue.groq_response_generator import (
                GroqResponseGenerator,
            )
            return GroqResponseGenerator()
        except ModuleNotFoundError:
            import logging
            logging.getLogger(__name__).warning(
                "GROQ_API_KEY is set but the optional 'groq' package is not installed; "
                "falling back to the lightweight rule-based generator."
            )

    from voicebot.dialogue.response_generator import (
        RuleBasedResponseGenerator,
    )
    return RuleBasedResponseGenerator()


class VoiceBotPipeline:
    """Ties the input, NLU, dialogue, and output stages together.

    Swap any one of these four components later without touching the
    others, e.g.:
        VoiceBotPipeline(
            input_source=WhisperInputSource(),      # real ASR
            intent_classifier=DistilBertClassifier(),  # real NLU
            response_generator=GroqResponseGenerator(),  # LLM responses
            output_sink=PiperOutputSink(),          # real TTS
        )
    """

    EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye"}

    def __init__(
        self,
        input_source=None,
        intent_classifier=None,
        response_generator=None,
        output_sink=None,
    ):
        self.input_source = input_source or ConsoleInputSource()
        self.intent_classifier = intent_classifier or KeywordIntentClassifier()
        self.response_generator = response_generator or _default_response_generator()
        self.output_sink = output_sink or ConsoleOutputSink()
        self.conversation_history = []

    def run_once(self, text: str):
        """Run one query through NLU -> dialogue -> output. Returns the
        Response object (useful for tests, without needing real I/O)."""
        intent = self.intent_classifier.classify(text)
        history = list(self.conversation_history)

        # If the generator supports user_text/history (Groq and enhanced rule-based
        # generators), pass them through; otherwise fall back to older signatures.
        try:
            response = self.response_generator.generate(
                intent,
                user_text=text,
                conversation_history=history,
            )
        except TypeError:
            try:
                response = self.response_generator.generate(intent, user_text=text)
            except TypeError:
                response = self.response_generator.generate(intent)

        self.conversation_history.append({"intent": intent.label, "text": text})
        self.output_sink.speak(response)
        return intent, response

    def run(self) -> None:
        """Main loop: read from input_source until exit or EOF."""
        print("AI Customer Support Voice Bot (text prototype)")
        print("Type your query below. Type 'exit' to quit.\n")

        while True:
            text = self.input_source.listen()

            if text is None:  # EOF, e.g. Ctrl+D
                print("\nSession ended.")
                break

            if not text:
                continue

            if text.lower() in self.EXIT_COMMANDS:
                print("Bot: Thanks for reaching out. Goodbye!")
                break

            self.run_once(text)


if __name__ == "__main__":
    # Load .env file (GROQ_API_KEY, etc.) before anything reads os.environ.
    from dotenv import load_dotenv
    load_dotenv()

    # Suppress noisy HTTP request logs from the groq/httpx libraries.
    import logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("groq").setLevel(logging.WARNING)

    VoiceBotPipeline().run()
