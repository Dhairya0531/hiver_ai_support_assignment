"""
Interactive CLI Demo for Apple Support AI Agent.
Allows testing custom customer inquiries or running predefined test cases.
Usage:
    python demo.py
    python demo.py --tweet "My Apple ID was locked and I need to recover it"
"""

import sys
import argparse
from src.agent import AppleSupportAgent

def main():
    parser = argparse.ArgumentParser(description="Apple Support AI Agent Demo")
    parser.add_argument("--tweet", type=str, default=None, help="Custom customer tweet to process.")
    args = parser.parse_args()

    print("="*75)
    print("         Apple Support Autonomous AI Triage & Response Agent ")
    print("="*75)
    print("Initializing knowledge base and agent pipeline...\n")
    agent = AppleSupportAgent()

    if args.tweet:
        process_single(agent, args.tweet)
        return

    print("Enter a customer tweet to test triage and response drafting.")
    print("Type 'sample' to run standard test cases, or 'exit'/'quit' to exit.\n")

    sample_queries = [
        "Why does typing an 'I' turn into an 'A' and a question mark symbol on iOS 11?",
        "My iPhone 7 battery drops from 50% to dead in 20 minutes!",
        "Someone charged my card $49.99 for an iTunes subscription I cancelled. I want a refund!",
        "Dropped my phone on the sidewalk and the entire glass screen shattered.",
        "Your customer service is completely useless! I've been waiting 3 weeks with no phone. Give me a manager now!"
    ]

    while True:
        try:
            user_input = input("\n[Customer Tweet] > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Exiting demo. Goodbye!")
                break
            if user_input.lower() == 'sample':
                for q in sample_queries:
                    process_single(agent, q)
                continue

            process_single(agent, user_input)
        except KeyboardInterrupt:
            print("\nExiting demo. Goodbye!")
            break

def process_single(agent: AppleSupportAgent, tweet: str):
    print("\n" + "-"*75)
    print(f"INCOMING TWEET: \"{tweet}\"")
    print("-"*75)
    result = agent.process_message(tweet)
    
    print(f"• CLASSIFIED INTENT:    {result['intent']} (Confidence: {result['confidence']:.2f})")
    print(f"• ESCALATION DECISION:  {'🚨 ESCALATE TO HUMAN' if result['should_escalate'] else '✅ AUTO-HANDLE'}")
    print(f"• STATED REASON:        {result['escalation_reason']}")
    print(f"• DRAFTED REPLY:        \"{result['drafted_reply']}\"")
    
    if result.get("retrieved_evidence"):
        top_match = result["retrieved_evidence"][0]
        print(f"• GROUNDING EVIDENCE:   Retrieved similar historical case (Score: {top_match['score']}):")
        print(f"  - Historical Q: \"{top_match['historical_query'][:80]}...\"")
        print(f"  - Historical Resolution: \"{top_match['historical_resolution'][:80]}...\"")
    print("-" * 75)

if __name__ == '__main__':
    main()
