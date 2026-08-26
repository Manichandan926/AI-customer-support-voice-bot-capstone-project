"""
dialogue/response_generator.py

Stands in for the dialogue management / response generation stage.
Right now it's a fixed lookup table of canned responses per intent.
Later, this is the file you replace with a real response-generation
model (e.g. an LLM or retrieval-augmented pipeline) — the interface
(generate() takes an Intent, returns a Response) stays the same.

This also implements the escalation rule from the project's system
design: if confidence is below a threshold, the bot hands off to a
human agent instead of guessing.
"""

from dataclasses import dataclass

from voicebot.nlu.intent_classifier import Intent


@dataclass
class Response:
    """A response ready to be spoken (or printed) back to the user."""
    text: str
    escalated: bool


class RuleBasedResponseGenerator:
    """Looks up a canned response for each supported intent.

    Below CONFIDENCE_THRESHOLD, the query is escalated to a human
    agent rather than answered — matching the escalation design in
    the project's system architecture (see Project_Roadmap.docx,
    Section 4.2, "Graceful escalation").
    """

    CONFIDENCE_THRESHOLD = 0.5

    RESPONSES = {
        "account_issue": (
            "I can help with account issues. To reset your password or "
            "unlock your account, please visit the 'Account Settings' "
            "page, or I can connect you to an agent for identity "
            "verification."
        ),
        "technical_support": (
            "I'm sorry you're running into a technical problem. Try "
            "restarting the device and reconnecting. If that doesn't "
            "resolve it, I can escalate this to our technical support team."
        ),
        "billing_inquiry": (
            "For billing questions, I can pull up your latest invoice "
            "or explain a charge. Could you confirm which billing cycle "
            "you're asking about?"
        ),
        "order_status": (
            "I can check your order status. Could you share your order "
            "number so I can look up the latest shipment update?"
        ),
        "general_inquiry": (
            "I'm not fully sure I understood that. Could you rephrase, "
            "or let me know if you'd like account, billing, technical, "
            "or order-status help?"
        ),
    }

    ESCALATION_MESSAGE = (
        "I want to make sure you get the right help here — let me "
        "connect you with a human agent for this one."
    )

    def generate(self, intent: Intent) -> Response:
        if intent.confidence < self.CONFIDENCE_THRESHOLD:
            return Response(text=self.ESCALATION_MESSAGE, escalated=True)

        text = self.RESPONSES.get(intent.label, self.RESPONSES["general_inquiry"])
        return Response(text=text, escalated=False)
