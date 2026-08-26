"""
input/console_input.py

Stands in for the ASR (speech-to-text) stage of the pipeline.
Right now it just reads text typed on the keyboard. Later, this is the
file you replace with a real ASR call (e.g. whisper.cpp) — everything
downstream only ever sees a plain string, so nothing else has to change.
"""

from typing import Optional


class ConsoleInputSource:
    """Mocked speech input source. Real implementation would capture
    audio from a microphone and run it through an ASR model."""

    def __init__(self, prompt: str = "You: "):
        self.prompt = prompt

    def listen(self) -> Optional[str]:
        """Return one user utterance as text, or None on EOF (Ctrl+D)."""
        try:
            text = input(self.prompt)
        except EOFError:
            return None
        return text.strip()
