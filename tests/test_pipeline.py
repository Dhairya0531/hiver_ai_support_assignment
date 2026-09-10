"""
Unit and Integration Tests for Apple Support AI Agent Pipeline.
"""

import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent import AppleSupportAgent
from src.taxonomy import ALL_INTENTS
from src.baselines import TrivialBaseline, SimpleBaseline

class TestAppleSupportAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = AppleSupportAgent()
        cls.trivial = TrivialBaseline()
        cls.simple = SimpleBaseline()

    def test_intent_coverage(self):
        """Test that agent intent output is always in ALL_INTENTS."""
        res = self.agent.process_message("My phone keeps freezing after updating iOS")
        self.assertIn(res["intent"], ALL_INTENTS)

    def test_security_escalation(self):
        """Account security and stolen devices MUST always escalate."""
        query = "My Apple ID was hacked and someone changed my password"
        res = self.agent.process_message(query)
        self.assertTrue(res["should_escalate"], "Account security breach must escalate!")
        self.assertEqual(res["reason_code"], "ACCOUNT_SECURITY")

    def test_billing_escalation(self):
        """Billing disputes and unauthorized charges MUST escalate."""
        query = "I was double charged $19.99 on iTunes for a subscription I canceled"
        res = self.agent.process_message(query)
        self.assertTrue(res["should_escalate"], "Billing disputes must escalate!")
        self.assertEqual(res["reason_code"], "BILLING_DISPUTE")

    def test_hardware_damage_escalation(self):
        """Physical hardware damage MUST escalate."""
        query = "My iPhone screen shattered on concrete and touch is unresponsive"
        res = self.agent.process_message(query)
        self.assertTrue(res["should_escalate"], "Shattered screen must escalate!")
        self.assertEqual(res["reason_code"], "HARDWARE_REPAIR")

    def test_customer_frustration_escalation(self):
        """Severe complaints and demand for human manager MUST escalate."""
        query = "This is the worst customer service ever, you stole my money and I demand a manager!"
        res = self.agent.process_message(query)
        self.assertTrue(res["should_escalate"], "Irate complaints must escalate!")
        self.assertEqual(res["reason_code"], "FRUSTRATED_CUSTOMER")

    def test_battery_autohandle(self):
        """Standard battery drain issues can be safely auto-handled."""
        query = "My iPhone 7 battery is draining fast after update, dies at 30%"
        res = self.agent.process_message(query)
        self.assertEqual(res["intent"], "battery_performance")
        self.assertFalse(res["should_escalate"])

    def test_rag_grounding_retrieval(self):
        """Verify that RAG returns non-empty historical evidence."""
        res = self.agent.process_message("How do I fix Bluetooth disconnecting from my car?")
        self.assertGreater(len(res["retrieved_evidence"]), 0)
        self.assertIn("historical_resolution", res["retrieved_evidence"][0])

    def test_baselines_execution(self):
        """Verify that both baselines run without crashing."""
        query = "Battery is draining rapidly"
        triv_res = self.trivial.predict(query)
        simp_res = self.simple.predict(query)
        self.assertIn("intent", triv_res)
        self.assertIn("intent", simp_res)

if __name__ == '__main__':
    unittest.main()
