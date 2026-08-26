"""
output/console_output.py

Stands in for the TTS (text-to-speech) stage of the pipeline. Right
now it just prints the response to the console. Later, this is the
file you replace with a real TTS call (e.g. piper-tts) — everything
upstream only ever produces a plain string, so nothing else has to
change.
"""

from voicebot.dialogue.response_generator import Response


class ConsoleOutputSink:
    """Mocked speech output sink. Real implementation would convert
    the response text to audio and play it back."""

    def __init__(self, prefix: str = "Bot: "):
        self.prefix = prefix

    def speak(self, response: Response) -> None:
        tag = " [ESCALATED]" if response.escalated else ""
        print(f"{self.prefix}{response.text}{tag}")
