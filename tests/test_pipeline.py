"""
tests/test_pipeline.py

Basic unit tests for the NLU and dialogue stages. Run with:
    python3 -m pytest
or, without pytest installed:
    python3 -m unittest discover
"""

import unittest

from voicebot.dialogue.response_generator import RuleBasedResponseGenerator
from voicebot.nlu.intent_classifier import KeywordIntentClassifier


class TestIntentClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = KeywordIntentClassifier()

    def test_account_issue_detected(self):
        intent = self.classifier.classify("I forgot my password and can't log in")
        self.assertEqual(intent.label, "account_issue")

    def test_technical_support_detected(self):
        intent = self.classifier.classify("My device keeps crashing after setup")
        self.assertEqual(intent.label, "technical_support")

    def test_billing_inquiry_detected(self):
        intent = self.classifier.classify("Why was I charged twice on my invoice?")
        self.assertEqual(intent.label, "billing_inquiry")

    def test_order_status_detected(self):
        intent = self.classifier.classify("Where is my package, has it shipped?")
        self.assertEqual(intent.label, "order_status")

    def test_unmatched_query_falls_back(self):
        intent = self.classifier.classify("asdkjaslkdj random gibberish")
        self.assertEqual(intent.label, "general_inquiry")
        self.assertEqual(intent.matched_keywords, [])

    def test_greeting_detected(self):
        intent = self.classifier.classify("hello there, I need help")
        self.assertEqual(intent.label, "greeting")

    def test_order_status_phrase_variation_detected(self):
        intent = self.classifier.classify("where is my order and when will it arrive?")
        self.assertEqual(intent.label, "order_status")


class TestResponseGenerator(unittest.TestCase):
    def setUp(self):
        self.classifier = KeywordIntentClassifier()
        self.generator = RuleBasedResponseGenerator()

    def test_high_confidence_intent_is_not_escalated(self):
        intent = self.classifier.classify("I need to reset my account password")
        response = self.generator.generate(intent)
        self.assertFalse(response.escalated)
        self.assertIn("account", response.text.lower())

    def test_low_confidence_query_is_escalated(self):
        intent = self.classifier.classify("asdkjaslkdj random gibberish")
        response = self.generator.generate(intent)
        self.assertTrue(response.escalated)

    def test_last_intent_context_is_used_for_fallback_follow_up(self):
        intent = self.classifier.classify("where is it?")
        response = self.generator.generate(
            intent,
            user_text="where is it?",
            conversation_history=[{"intent": "order_status"}],
        )
        self.assertIn("order status", response.text.lower())


if __name__ == "__main__":
    unittest.main()
