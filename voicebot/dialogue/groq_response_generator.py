"""
dialogue/groq_response_generator.py

LLM-powered response generator using the Groq API.  Replaces the
rule-based canned-response lookup with a real language model call,
while keeping the exact same interface (generate() takes an Intent,
returns a Response) so the rest of the pipeline is unaffected.

Requires:
    pip install groq
    export GROQ_API_KEY="gsk_..."
"""

import os
import re
from typing import List, Optional

from groq import Groq

from voicebot.dialogue.response_generator import Response
from voicebot.nlu.intent_classifier import Intent


SYSTEM_PROMPT = """\
/no_think
You are a friendly, professional AI customer-support agent for a
technology company.  Your job is to help users with:

  - Account issues (password resets, login problems, profile changes)
  - Technical support (device errors, connectivity, setup)
  - Billing inquiries (invoices, charges, refunds, subscriptions)
  - Order status (shipment tracking, delivery updates)
  - General queries and greetings

Rules you MUST follow:
  1. Be concise — keep replies to 2-3 sentences max.
  2. Never fabricate order numbers, tracking IDs, or account details.
  3. If you truly cannot help, offer to connect the user to a human agent.
  4. Stay on topic — politely decline off-topic requests.
  5. Be empathetic and solution-oriented.
  6. Reply directly with ONLY the response text — no thinking, no tags.
"""


class GroqResponseGenerator:
    """Generates responses via the Groq LLM API.

    Drop-in replacement for ``RuleBasedResponseGenerator``.  Every user
    message is sent to the LLM — the model decides whether it can help
    or needs to escalate, rather than relying on keyword confidence.
    """

    def __init__(
        self,
        model: str = "openai/gpt-oss-20b",
        api_key: Optional[str] = None,
        temperature: float = 0.4,
        max_tokens: int = 256,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Groq API key is required. Either pass api_key= or set "
                "the GROQ_API_KEY environment variable."
            )
        self.client = Groq(api_key=resolved_key)

        # Conversation history for multi-turn context.
        self._history: List[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

    def generate(self, intent: Intent, user_text: str = "") -> Response:
        """Generate a response for the user query using the Groq LLM.

        All queries go to the LLM — it handles greetings, ambiguous
        messages, and everything in between.
        """
        prompt = user_text if user_text else intent.label
        self._history.append({"role": "user", "content": prompt})

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self._history,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            reply = completion.choices[0].message.content.strip()
            # Strip <think>…</think> reasoning blocks if present.
            reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL)
            reply = re.sub(r"<think>.*", "", reply, flags=re.DOTALL)
            reply = reply.strip()
        except Exception:
            reply = (
                "I'm having a little trouble on my end right now. "
                "Let me connect you with a human agent instead."
            )
            return Response(text=reply, escalated=True)

        self._history.append({"role": "assistant", "content": reply})
        return Response(text=reply, escalated=False)

    def reset_conversation(self) -> None:
        """Clear conversation history (keeps only the system prompt)."""
        self._history = [{"role": "system", "content": SYSTEM_PROMPT}]
