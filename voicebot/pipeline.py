"""
pipeline.py

Wires the four pipeline stages together:

    input (ASR)  ->  nlu (intent)  ->  dialogue (response)  ->  output (TTS)

Each stage is a small, swappable object. To go from this text-only
prototype to the real voice pipeline, you only ever replace the
class passed in here for one stage — the VoiceBotPipeline class
itself does not change.
"""

from voicebot.dialogue.response_generator import RuleBasedResponseGenerator
from voicebot.input.console_input import ConsoleInputSource
from voicebot.nlu.intent_classifier import KeywordIntentClassifier
from voicebot.output.console_output import ConsoleOutputSink


class VoiceBotPipeline:
    """Ties the input, NLU, dialogue, and output stages together.

    Swap any one of these four components later without touching the
    others, e.g.:
        VoiceBotPipeline(
            input_source=WhisperInputSource(),      # real ASR
            intent_classifier=DistilBertClassifier(),  # real NLU
            response_generator=RuleBasedResponseGenerator(),  # unchanged
            output_sink=PiperOutputSink(),          # real TTS
        )
    """

    EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye"}

    def __init__(
        self,
        input_source: ConsoleInputSource = None,
        intent_classifier: KeywordIntentClassifier = None,
        response_generator: RuleBasedResponseGenerator = None,
        output_sink: ConsoleOutputSink = None,
    ):
        self.input_source = input_source or ConsoleInputSource()
        self.intent_classifier = intent_classifier or KeywordIntentClassifier()
        self.response_generator = response_generator or RuleBasedResponseGenerator()
        self.output_sink = output_sink or ConsoleOutputSink()

    def run_once(self, text: str):
        """Run one query through NLU -> dialogue -> output. Returns the
        Response object (useful for tests, without needing real I/O)."""
        intent = self.intent_classifier.classify(text)
        response = self.response_generator.generate(intent)
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
    VoiceBotPipeline().run()
