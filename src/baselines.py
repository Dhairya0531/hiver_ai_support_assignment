"""
Baseline models for customer support triage and response generation.
Required Deliverable: Comparison vs. at least two baselines:
1. Trivial Baseline: Majority class intent + static canned response + default auto-handle
2. Simple Baseline: Lexical 1-NN retrieval without prompt grounding + keyword heuristic triage
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class TrivialBaseline:
    """
    Baseline 1: Trivial Heuristic
    - Intent: Always predicts majority class ('software_update_os')
    - Escalation: Always False (never escalates)
    - Reply: Static canned template
    """
    def __init__(self, majority_intent: str = "software_update_os"):
        self.majority_intent = majority_intent
        self.canned_reply = "Thanks for reaching out to Apple Support. We'd like to help you with this. Please send us a DM with your device model and iOS version: https://t.co/GDrqU22YpT"

    def predict(self, customer_text: str) -> Dict[str, Any]:
        return {
            "intent": self.majority_intent,
            "confidence": 0.50,
            "should_escalate": False,
            "reason_code": "AUTO_HANDLE",
            "escalation_reason": "Trivial baseline default auto-handle assumption.",
            "drafted_reply": self.canned_reply
        }

class SimpleBaseline:
    """
    Baseline 2: Simple TF-IDF 1-NN + Heuristic Triage
    - Intent: TF-IDF similarity to intent definition keywords
    - Escalation: Simple regex keyword match (refund, broken, locked, agent)
    - Reply: Direct 1-Nearest-Neighbor historical tweet (verbatim copy without re-drafting)
    """
    def __init__(self, sample_data_path: str = 'data/processed/applesupport_pairs_sample.csv'):
        self.df = pd.read_csv(sample_data_path).head(5000)
        self.df['customer_text'] = self.df['customer_text'].astype(str)
        self.df['agent_text'] = self.df['agent_text'].astype(str)
        
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['customer_text'])
        
        self.escalation_keywords = [
            "refund", "charge", "charged", "money", "stolen", "broken",
            "shattered", "screen", "locked", "password", "human", "agent", "manager", "lawyer"
        ]

    def predict(self, customer_text: str) -> Dict[str, Any]:
        text_lower = customer_text.lower()
        
        # 1. Simple heuristic escalation
        should_escalate = any(k in text_lower for k in self.escalation_keywords)
        reason_code = "HARDWARE_OR_BILLING_KEYWORD" if should_escalate else "AUTO_HANDLE"
        
        # 2. Simple keyword intent rule
        if any(w in text_lower for w in ["battery", "drain", "dying"]):
            intent = "battery_performance"
        elif any(w in text_lower for w in ["screen", "broken", "cracked", "repair"]):
            intent = "hardware_repair_service"
        elif any(w in text_lower for w in ["refund", "charge", "double billed"]):
            intent = "billing_subscriptions"
        elif any(w in text_lower for w in ["apple id", "password", "locked", "stolen"]):
            intent = "account_apple_id_security"
        elif any(w in text_lower for w in ["bluetooth", "wifi", "airpods"]):
            intent = "connectivity_peripherals"
        elif any(w in text_lower for w in ["worst", "terrible", "manager", "human"]):
            intent = "escalation_human_complaint"
        elif any(w in text_lower for w in ["ios", "update", "freeze", "crash"]):
            intent = "software_update_os"
        else:
            intent = "general_guidance"

        # 3. 1-NN verbatim reply retrieval
        query_vec = self.vectorizer.transform([customer_text])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_idx = int(np.argmax(sims))
        verbatim_reply = self.df.iloc[top_idx]['agent_text']
        
        return {
            "intent": intent,
            "confidence": float(sims[top_idx]),
            "should_escalate": should_escalate,
            "reason_code": reason_code,
            "escalation_reason": "Escalated based on simple keyword heuristic." if should_escalate else "Auto-handled by heuristic.",
            "drafted_reply": verbatim_reply
        }

if __name__ == '__main__':
    t_base = TrivialBaseline()
    s_base = SimpleBaseline()
    
    sample_query = "My screen is completely shattered and won't turn on"
    print("Query:", sample_query)
    print("\nTrivial Baseline:", t_base.predict(sample_query))
    print("\nSimple Baseline:", s_base.predict(sample_query))
