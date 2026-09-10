"""
AppleSupport AI Support Agent Pipeline.
Coordinates:
1. Intent Classification (8 MECE intents)
2. Grounded Historical Retrieval (RAG)
3. Escalation & Triage Decision Engine with stated policy reasons
4. Reply Drafting adhering to official Apple brand voice
"""

import json
from typing import Dict, Any, Optional

try:
    from src.taxonomy import INTENT_DEFINITIONS, ALL_INTENTS, ESCALATION_REASONS
    from src.rag import SupportKnowledgeBase
    from src.llm_client import GeminiLLMClient
except ImportError:
    from taxonomy import INTENT_DEFINITIONS, ALL_INTENTS, ESCALATION_REASONS  # type: ignore
    from rag import SupportKnowledgeBase  # type: ignore
    from llm_client import GeminiLLMClient  # type: ignore

CLASSIFIER_PROMPT_TEMPLATE = """You are the Lead Support Triage AI for Apple Support on Twitter.
Analyze the following incoming customer tweet and perform two tasks:
1. Classify the tweet into EXACTLY ONE of the following 8 intent categories:
{intents_description}

2. Decide whether this message should be AUTO-HANDLED or ESCALATED TO A HUMAN AGENT:
Policy Rules for Escalation:
- ESCALATE if the inquiry involves account credentials, locked Apple ID, 2FA, or stolen devices (ACCOUNT_SECURITY).
- ESCALATE if the inquiry involves unauthorized charges, double billing, or refund disputes (BILLING_DISPUTE).
- ESCALATE if the inquiry involves cracked screens, liquid damage, or physical hardware failure requiring Genius Bar repair (HARDWARE_REPAIR).
- ESCALATE if the customer exhibits extreme frustration, uses abusive language, threatens legal action, or demands a human/supervisor (FRUSTRATED_CUSTOMER).
- ESCALATE if the query is extremely vague, unanswerable, or poses high hallucination risk (LOW_CONFIDENCE).
- AUTO-HANDLE if it is a standard software bug, settings inquiry, battery optimization, connectivity reset, or general how-to that can be guided via public documentation and safe troubleshooting steps.

Customer Tweet:
"{customer_text}"

You must respond in valid JSON format only, with the following keys:
{{
  "intent": "<one of the 8 intent ids>",
  "confidence": <float between 0.0 and 1.0>,
  "should_escalate": <true or false>,
  "reason_code": "<one of ACCOUNT_SECURITY, BILLING_DISPUTE, HARDWARE_REPAIR, FRUSTRATED_CUSTOMER, LOW_CONFIDENCE, AUTO_HANDLE>",
  "escalation_reason": "<clear explanation of why this was escalated or auto-handled>"
}}
"""

REPLY_PROMPT_TEMPLATE = """You are an official Apple Support representative on Twitter (@AppleSupport).
Draft a helpful, professional, and empathetic tweet reply to the customer.

Brand Tone & Twitter Guidelines:
- Keep the reply concise, polite, and reassuring (maximum 2-3 sentences, Twitter style).
- If AUTO-HANDLED: Provide actionable first-line troubleshooting steps or direct to official Apple settings/documentation (e.g., Settings > General, force restart).
- If ESCALATED: Express empathy, explain that their issue requires private review (or an appointment), and direct them safely (e.g. invite to DM or link to official portal like iforgot.apple.com or reportaproblem.apple.com).
- NEVER ask the customer to post personal account details or passwords publicly.
- Ground your reply in the historical evidence below.

CUSTOMER INQUIRY:
"{customer_text}"

TRIAGE DECISION:
- Intent: {intent}
- Should Escalate: {should_escalate}
- Escalation Reason: {escalation_reason}

{grounding_context}

Draft the tweet reply:"""

class AppleSupportAgent:
    def __init__(self, kb: Optional[SupportKnowledgeBase] = None, llm: Optional[GeminiLLMClient] = None):
        self.kb = kb or SupportKnowledgeBase()
        self.llm = llm or GeminiLLMClient()

    def _format_intents_for_prompt(self) -> str:
        lines = []
        for i_id, idef in INTENT_DEFINITIONS.items():
            lines.append(f"- {i_id}: {idef.display_name}. {idef.description}")
        return "\n".join(lines)

    def classify_and_triage(self, customer_text: str) -> Dict[str, Any]:
        """
        Classifies intent and decides escalation status with stated rationale.
        """
        prompt = CLASSIFIER_PROMPT_TEMPLATE.format(
            intents_description=self._format_intents_for_prompt(),
            customer_text=customer_text
        )
        
        system_instruction = "You are an expert AI triage classifier for Apple Support. Return only structured JSON."
        result = self.llm.generate_json(prompt, system_instruction=system_instruction)
        
        # Validate intent
        intent = result.get("intent", "general_guidance")
        if intent not in ALL_INTENTS:
            intent = "general_guidance"
            
        reason_code = result.get("reason_code", "AUTO_HANDLE")
        should_escalate = bool(result.get("should_escalate", False))
        
        # Policy enforcement safeguard: If intent requires escalation by security policy, enforce it
        if intent in ["account_apple_id_security", "billing_subscriptions", "hardware_repair_service", "escalation_human_complaint"]:
            should_escalate = True
            if reason_code == "AUTO_HANDLE":
                if intent == "account_apple_id_security": reason_code = "ACCOUNT_SECURITY"
                elif intent == "billing_subscriptions": reason_code = "BILLING_DISPUTE"
                elif intent == "hardware_repair_service": reason_code = "HARDWARE_REPAIR"
                elif intent == "escalation_human_complaint": reason_code = "FRUSTRATED_CUSTOMER"

        escalation_reason = result.get("escalation_reason") or ESCALATION_REASONS.get(reason_code, "Policy-driven triage.")
        confidence = float(result.get("confidence", 0.90))

        return {
            "intent": intent,
            "confidence": confidence,
            "should_escalate": should_escalate,
            "reason_code": reason_code,
            "escalation_reason": escalation_reason
        }

    def draft_reply(self, customer_text: str, triage: Dict[str, Any], retrieved_examples: list) -> str:
        """
        Drafts grounded response using retrieved historical resolutions.
        """
        grounding_context = self.kb.format_grounding_context(retrieved_examples)
        prompt = REPLY_PROMPT_TEMPLATE.format(
            customer_text=customer_text,
            intent=triage["intent"],
            should_escalate=triage["should_escalate"],
            escalation_reason=triage["escalation_reason"],
            grounding_context=grounding_context
        )
        reply = self.llm.generate_text(prompt)
        return reply

    def process_message(self, customer_text: str) -> Dict[str, Any]:
        """
        End-to-end execution on an incoming customer tweet.
        """
        # 1. RAG retrieval
        retrieved = self.kb.retrieve_similar_resolutions(customer_text, top_k=3)
        
        # 2. Intent & Escalation Triage
        triage = self.classify_and_triage(customer_text)
        
        # 3. Grounded Reply Drafting
        reply = self.draft_reply(customer_text, triage, retrieved)
        
        return {
            "customer_text": customer_text,
            "intent": triage["intent"],
            "confidence": triage["confidence"],
            "should_escalate": triage["should_escalate"],
            "reason_code": triage["reason_code"],
            "escalation_reason": triage["escalation_reason"],
            "drafted_reply": reply,
            "retrieved_evidence": retrieved
        }

if __name__ == '__main__':
    agent = AppleSupportAgent()
    test_cases = [
        "Why does typing an 'I' turn into an 'A' and a strange question mark box on my iPhone?",
        "My iPhone 7 battery is dying at 30% in less than 2 hours!",
        "Someone made 3 unauthorized charges on my credit card from the App Store. I need a refund right now!",
        "My phone screen shattered when I dropped it on concrete. Can I book a repair?",
        "This is the worst customer service ever. You stole my money and ignored my messages. I want a manager immediately!"
    ]
    for test in test_cases:
        print(f"\n==========================================")
        print(f"CUSTOMER: {test}")
        res = agent.process_message(test)
        print(f"INTENT: {res['intent']} (Conf: {res['confidence']})")
        print(f"ESCALATE: {res['should_escalate']} | REASON: {res['escalation_reason']}")
        print(f"DRAFTED REPLY: {res['drafted_reply']}")
