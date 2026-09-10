"""
Central Evaluation Harness for Apple Support AI Agent & Baselines.
Compares:
1. Baseline 1: Trivial Baseline (Majority intent, default auto-handle, canned reply)
2. Baseline 2: Simple Baseline (1-NN retrieval, keyword triage)
3. Proposed Agent: Grounded RAG + Policy-calibrated LLM Triage + Apple Voice Reply

Measures:
- Intent Classification (Accuracy, Macro F1)
- Escalation Triage (Precision, Recall, F1, False Auto-Handle Rate)
- Reply Quality (Groundedness, Helpfulness, Tone, Safety, Overall via LLM Judge)
- Human vs. Judge Calibration (Pearson r, Spearman rho, MAE, Quadratic Cohen's Kappa)
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
from tqdm import tqdm

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.taxonomy import ALL_INTENTS
from src.agent import AppleSupportAgent
from src.baselines import TrivialBaseline, SimpleBaseline
from src.llm_client import GeminiLLMClient
from evaluation.judge import ReplyQualityJudge
from evaluation.metrics import compute_intent_metrics, compute_escalation_metrics, compute_human_judge_agreement

RESULTS_DIR = "evaluation/results"

def run_evaluation(num_samples: int = None, use_live_judge: bool = True):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    golden_path = "data/golden/golden_set_200.json"
    
    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"Golden dataset not found at {golden_path}. Run build_golden_set.py first.")
        
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)
        
    if num_samples and num_samples < len(golden_data):
        print(f"[Eval] Running quick evaluation on first {num_samples} golden examples...")
        eval_set = golden_data[:num_samples]
    else:
        print(f"[Eval] Running full evaluation on all {len(golden_data)} golden examples...")
        eval_set = golden_data

    # Initialize models
    print("[Eval] Initializing models and judges...")
    trivial_model = TrivialBaseline()
    simple_model = SimpleBaseline()
    agent = AppleSupportAgent()
    judge = ReplyQualityJudge()

    systems = {
        "Baseline 1 (Trivial)": trivial_model,
        "Baseline 2 (Simple)": simple_model,
        "Proposed Agent (RAG + Triage)": agent
    }

    results = {
        "models": {},
        "human_judge_calibration": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_count": len(eval_set)
    }

    detailed_rows = []
    human_scores_for_calibration = []
    judge_scores_on_human_replies = []

    # Run predictions
    for model_name, model in systems.items():
        print(f"\n--- Evaluating {model_name} ---")
        y_true_intent = []
        y_pred_intent = []
        y_true_esc = []
        y_pred_esc = []
        
        groundedness_scores = []
        helpfulness_scores = []
        tone_scores = []
        safety_scores = []
        overall_scores = []
        
        start_time = time.time()
        
        for idx, item in enumerate(tqdm(eval_set, desc=model_name)):
            c_text = item["customer_text"]
            true_intent = item["true_intent"]
            true_esc = item["true_should_escalate"]
            key_pts = item["key_resolution_points"]
            
            # Predict
            if hasattr(model, "process_message"):
                pred = model.process_message(c_text)
            else:
                pred = model.predict(c_text)
                
            y_true_intent.append(true_intent)
            y_pred_intent.append(pred["intent"])
            y_true_esc.append(true_esc)
            y_pred_esc.append(pred["should_escalate"])

            # Run LLM-as-a-judge on reply quality
            # Sample every 5th for deep judge scoring if quick, or evaluate all
            judge_res = judge.evaluate_reply(
                customer_text=c_text,
                agent_reply=pred["drafted_reply"],
                key_resolution_points=key_pts,
                intent=pred["intent"],
                should_escalate=pred["should_escalate"]
            )
            groundedness_scores.append(judge_res["groundedness"])
            helpfulness_scores.append(judge_res["helpfulness"])
            tone_scores.append(judge_res["tone"])
            safety_scores.append(judge_res["safety"])
            overall_scores.append(judge_res["overall_score"])

            # If proposed agent, also judge historical human reply for calibration
            if model_name == "Proposed Agent (RAG + Triage)":
                h_reply = item["historical_agent_reply"]
                h_judge = judge.evaluate_reply(
                    customer_text=c_text,
                    agent_reply=h_reply,
                    key_resolution_points=key_pts,
                    intent=true_intent,
                    should_escalate=true_esc
                )
                human_scores_for_calibration.append(item["human_quality_score"])
                judge_scores_on_human_replies.append(h_judge["overall_score"])

            detailed_rows.append({
                "example_id": item["id"],
                "model": model_name,
                "customer_text": c_text,
                "true_intent": true_intent,
                "pred_intent": pred["intent"],
                "true_escalate": true_esc,
                "pred_escalate": pred["should_escalate"],
                "reason_code": pred.get("reason_code", ""),
                "escalation_reason": pred.get("escalation_reason", ""),
                "drafted_reply": pred["drafted_reply"],
                "judge_groundedness": judge_res["groundedness"],
                "judge_helpfulness": judge_res["helpfulness"],
                "judge_tone": judge_res["tone"],
                "judge_safety": judge_res["safety"],
                "judge_overall": judge_res["overall_score"]
            })

        elapsed_time = round(time.time() - start_time, 2)
        avg_latency_ms = round((elapsed_time / len(eval_set)) * 1000, 1)

        # Compute metrics
        intent_metrics = compute_intent_metrics(y_true_intent, y_pred_intent, ALL_INTENTS)
        esc_metrics = compute_escalation_metrics(y_true_esc, y_pred_esc)

        results["models"][model_name] = {
            "intent": intent_metrics,
            "escalation": esc_metrics,
            "reply_quality": {
                "groundedness": round(float(pd.Series(groundedness_scores).mean()), 2),
                "helpfulness": round(float(pd.Series(helpfulness_scores).mean()), 2),
                "tone": round(float(pd.Series(tone_scores).mean()), 2),
                "safety": round(float(pd.Series(safety_scores).mean()), 2),
                "overall_quality": round(float(pd.Series(overall_scores).mean()), 2)
            },
            "latency": {
                "total_seconds": elapsed_time,
                "avg_ms_per_query": avg_latency_ms
            }
        }

    # Human-Judge Agreement Calibration
    calibration = compute_human_judge_agreement(human_scores_for_calibration, judge_scores_on_human_replies)
    results["human_judge_calibration"] = calibration

    # Save detailed outputs
    with open(os.path.join(RESULTS_DIR, "evaluation_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    df_details = pd.DataFrame(detailed_rows)
    df_details.to_csv(os.path.join(RESULTS_DIR, "detailed_predictions.csv"), index=False)

    # Print Headline Comparison Table
    print("\n" + "="*85)
    print("                      HEADLINE BENCHMARK COMPARISON")
    print("="*85)
    print(f"{'Metric':<35} | {'Baseline 1 (Trivial)':<20} | {'Baseline 2 (Simple)':<20} | {'Proposed Agent':<15}")
    print("-" * 85)
    
    m_triv = results["models"]["Baseline 1 (Trivial)"]
    m_simp = results["models"]["Baseline 2 (Simple)"]
    m_agent = results["models"]["Proposed Agent (RAG + Triage)"]
    
    print(f"{'Intent Accuracy':<35} | {m_triv['intent']['accuracy']*100:>19.1f}% | {m_simp['intent']['accuracy']*100:>19.1f}% | {m_agent['intent']['accuracy']*100:>14.1f}%")
    print(f"{'Intent Macro F1':<35} | {m_triv['intent']['macro_f1']*100:>19.1f}% | {m_simp['intent']['macro_f1']*100:>19.1f}% | {m_agent['intent']['macro_f1']*100:>14.1f}%")
    print(f"{'Escalation Accuracy':<35} | {m_triv['escalation']['accuracy']*100:>19.1f}% | {m_simp['escalation']['accuracy']*100:>19.1f}% | {m_agent['escalation']['accuracy']*100:>14.1f}%")
    print(f"{'Escalation Recall':<35} | {m_triv['escalation']['recall']*100:>19.1f}% | {m_simp['escalation']['recall']*100:>19.1f}% | {m_agent['escalation']['recall']*100:>14.1f}%")
    print(f"{'Escalation Precision':<35} | {m_triv['escalation']['precision']*100:>19.1f}% | {m_simp['escalation']['precision']*100:>19.1f}% | {m_agent['escalation']['precision']*100:>14.1f}%")
    print(f"{'False Auto-Handle Rate (Risk)':<35} | {m_triv['escalation']['false_auto_handle_rate']*100:>19.1f}% | {m_simp['escalation']['false_auto_handle_rate']*100:>19.1f}% | {m_agent['escalation']['false_auto_handle_rate']*100:>14.1f}%")
    print(f"{'Judge Reply Quality (1-5)':<35} | {m_triv['reply_quality']['overall_quality']:>20.2f} | {m_simp['reply_quality']['overall_quality']:>20.2f} | {m_agent['reply_quality']['overall_quality']:>15.2f}")
    print(f"{'Avg Latency per Query':<35} | {m_triv['latency']['avg_ms_per_query']:>18.1f}ms | {m_simp['latency']['avg_ms_per_query']:>18.1f}ms | {m_agent['latency']['avg_ms_per_query']:>13.1f}ms")
    print("="*85)

    print("\n" + "="*60)
    print("     LLM-AS-A-JUDGE HUMAN AGREEMENT CALIBRATION")
    print("="*60)
    print(f"Mean Absolute Error (MAE):       {calibration['mae']}")
    print(f"Exact Agreement:                 {calibration['exact_agreement_pct']}%")
    print(f"Adjacent Agreement (+-1 score):  {calibration['adjacent_agreement_pct']}%")
    print(f"Pearson Correlation (r):         {calibration['pearson_correlation_r']}")
    print(f"Spearman Rank Correlation (rho): {calibration['spearman_rank_rho']}")
    print(f"Cohen's Kappa (Quadratic):       {calibration['cohen_kappa_quadratic']}")
    print("="*60)
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Apple Support Agent against Baselines.")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to evaluate (default: all 200).")
    args = parser.parse_args()
    
    run_evaluation(num_samples=args.samples)
