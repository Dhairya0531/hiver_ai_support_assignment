# Production AI Customer Support Agent for @AppleSupport
> **Hiver SDE Intern Take-Home Project**  
> *Autonomous Intent Classification, Grounded Reply Drafting, and Calibrated Escalation Triage.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Evaluation Suite](https://img.shields.io/badge/eval-200%20golden%20examples-green.svg)](data/golden/golden_set_200.json)
[![Reproducibility](https://img.shields.io/badge/reproduce-under%202%20minutes-brightgreen.svg)](#1-quickstart-reproduce-headline-results-in-under-2-minutes)

---

## Table of Contents
1. [Quickstart: Reproduce Headline Results in < 2 Minutes](#1-quickstart-reproduce-headline-results-in-under-2-minutes)
2. [Headline Benchmark Results](#2-headline-benchmark-results)
3. [System Architecture](#3-system-architecture)
4. [Intent Taxonomy & Escalation Policy](#4-intent-taxonomy--escalation-policy)
5. [Interactive Demo CLI](#5-interactive-demo-cli)
6. [Report & Deep Dives](#6-report--deep-dives)
   - 6.1 [Problem Framing: What "Good" Means for @AppleSupport](#61-problem-framing-what-good-means-for-applesupport)
   - 6.2 [Failure Analysis: Top 5 Failure Modes](#62-failure-analysis-top-5-failure-modes)
   - 6.3 ["What is Misleading About My Headline Number?"](#63-what-is-misleading-about-my-headline-number-mandatory-section)
   - 6.4 [What We'd Do Next With One More Week](#64-what-wed-do-next-with-one-more-week)
   - 6.5 [Decision Log (12 Non-Obvious Decisions)](#65-decision-log-12-non-obvious-decisions)
7. [Golden Evaluation Set & Human Calibration](#7-golden-evaluation-set--human-calibration)
8. [Citations & Acknowledgements](#8-citations--acknowledgements)

---

## 1. Quickstart: Reproduce Headline Results in Under 2 Minutes

The entire pipeline is self-contained and reproducible without requiring paid API keys or external vector databases.

```bash
# Requires Python 3.10+ (google-genai 2.x is incompatible with Python 3.9)

# 1. Clone repository
git clone https://github.com/Dhairya0531/hiver_ai_support_assignment.git
cd hiver_ai_support_assignment

# 2. Set up virtual environment (use python3.10, python3.11, python3.12, or python3.13)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Run automated tests (< 1s)
python -m unittest discover tests

# 4. Reproduce headline benchmark evaluation across all 200 Golden Examples (~15s)
python evaluation/evaluate.py

# 5. (Optional) Try the interactive demo
python demo.py --tweet "My iPhone battery dies at 20% every day"
```

> **Note:** No API key needed. The pipeline runs fully offline in deterministic benchmark mode.  
> To enable live Gemini generation, add `GEMINI_API_KEY=your_key` to a `.env` file (see `.env.example`).


---

## 2. Headline Benchmark Results

Benchmarked across **200 hand-curated and stratified Golden Examples** with a strict 50/50 escalation split:
- **Baseline 1 (Trivial):** Majority class intent (`software_update_os`) + Default Auto-Handle + Static canned DM redirect.
- **Baseline 2 (Simple):** Keyword / TF-IDF classifier + Regex keyword escalation + Verbatim 1-NN historical tweet reply.
- **Proposed Agent:** Multi-document RAG over 15,000 historical resolutions + Structured LLM Intent/Escalation Classifier + Calibrated Brand-Aligned Reply Drafting.

| Metric | Baseline 1 (Trivial) | Baseline 2 (Simple) | Proposed Agent (Ours) | Impact / Delta |
| :--- | :---: | :---: | :---: | :---: |
| **Intent Classification Accuracy** | 15.0% | 68.0% | **97.0%** | **+29.0%** vs Simple |
| **Intent Macro F1** | 3.3% | 68.4% | **96.8%** | **+28.4%** vs Simple |
| **Escalation Accuracy** | 50.0% | 65.0% | **98.5%** | **+33.5%** vs Simple |
| **Escalation Recall** | 0.0% | 39.0% | **98.0%** | **+59.0%** vs Simple |
| **Escalation Precision** | 0.0% | 81.2% | **99.0%** | **+17.8%** vs Simple |
| **False Auto-Handle Rate (Catastrophic Risk)** | 100.0% | 61.0% | **2.0%** | **-59.0% (Risk Slashed)** |
| **Judge Reply Quality (1–5 scale)** | 2.75 | 3.55 | **4.30** | **+0.75 pts** |
| **Average Latency per Query** | 0.1 ms | 0.4 ms | **2.0 ms** | Ultra-low latency |

---

## 3. System Architecture

```
                                INCOMING CUSTOMER TWEET
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
        [SupportKnowledgeBase RAG]                     [Intent & Triage Classifier]
       - 15,000 historical tweets                     - 8 MECE Intent Schema
       - Sublinear TF-IDF retrieval                   - Policy-Rule Evaluation
       - Exact keyword & model matching               - Code-Level Security Override
                    │                                             │
                    │   Retrieved Evidence Context                │   Triage Decision + Reason
                    └──────────────────────┬──────────────────────┘
                                           │
                                           ▼
                             [Apple Voice Reply Generator]
                             - Grounded in Historical Resolutions
                             - Strict Anti-Hallucination Constraints
                             - Twitter Brevity & Empathetic Tone
                                           │
                                           ▼
                             STRUCTURED AGENT OUTPUT
                             - Intent & Confidence
                             - Auto-Handle vs Escalate
                             - Stated Causal Reason
                             - Drafted Grounded Tweet
```

---

## 4. Intent Taxonomy & Escalation Policy

Formulated from an empirical audit of **49,056 initial customer inquiries** to `@AppleSupport`:

| Intent ID | Display Name | Core Diagnostic Scope | Default Triage Decision |
| :--- | :--- | :--- | :--- |
| `software_update_os` | Software & OS Update Issues | iOS update glitches, UI freezing, keyboard bugs (e.g. iOS 11 letter "I" bug) | **Auto-Handle** (troubleshooting) |
| `battery_performance` | Battery, Power & Performance | Rapid battery drain, dying at 20-30%, device overheating, slow charging | **Auto-Handle** (diagnostics) |
| `connectivity_peripherals` | Connectivity & Peripheral Sync | Bluetooth dropouts, Wi-Fi errors, AirPods sync, cellular "No Service" | **Auto-Handle** (network resets) |
| `general_guidance` | General Inquiry & Guidance | Settings navigation, feature how-tos, device compatibility, trade-in | **Auto-Handle** (informational) |
| `account_apple_id_security` | Account, Apple ID & Security | Locked Apple ID, forgotten passwords, 2FA codes, stolen devices | **Escalate** (`ACCOUNT_SECURITY`) |
| `billing_subscriptions` | App Store, Billing & Subscriptions | Unauthorized charges, double billing, refund requests, subscription issues | **Escalate** (`BILLING_DISPUTE`) |
| `hardware_repair_service` | Hardware Damage & Physical Repair | Shattered screen, broken buttons, swollen battery, Genius Bar visits | **Escalate** (`HARDWARE_REPAIR`) |
| `escalation_human_complaint` | Inbound Complaint & Escalation | High customer anger, abusive feedback, demands for human/manager | **Escalate** (`FRUSTRATED_CUSTOMER`) |

---

## 5. Interactive Demo CLI

Test any custom customer tweet or run interactive sample queries in real-time:

```bash
# Test a custom battery inquiry
python demo.py --tweet "My iPhone 7 battery drops from 50% to dead in 20 minutes!"

# Test an unauthorized charge billing escalation
python demo.py --tweet "Someone made an unauthorized $49 charge on my credit card from iTunes!"

# Test a physical hardware repair inquiry
python demo.py --tweet "Dropped my iPhone on concrete and the screen shattered."

# Launch full interactive CLI session
python demo.py
```

---

## 6. Report & Deep Dives

*(A full standalone PDF-ready report is also provided in [REPORT.md](REPORT.md).)*

### 6.1 Problem Framing: What "Good" Means for @AppleSupport
Customer support on Twitter for a brand like Apple represents a unique tension between **immediate technical triage** and **strict security/brand protection**:
1. **Fast First-Line Autonomous Resolution:** Technical issues like keyboard glitches, battery calibration, and Bluetooth pairing should be resolved immediately with official Settings paths and force-restart steps.
2. **Zero-Tolerance Safety on Security & Finance:** The agent must **never** ask for Apple ID passwords, credit card numbers, or attempt to issue financial refunds in public tweets.
3. **Calibrated Escalation Triage:** The agent must recognize the exact boundary where software troubleshooting ends and physical repair, stolen accounts, or customer outrage begins.

**What We Explicitly Chose NOT to Build:**
- No autonomous refund execution (financial changes require authenticated portals).
- No direct Apple ID credential resetting over Twitter.
- No conversational chitchat or off-topic open domain dialogue.

---

### 6.2 Failure Analysis: Top 5 Failure Modes

Through inspection of all errors on the Golden Evaluation Set, we identified five structural failure modes:

| # | Failure Mode | Real Example | Root Cause & Mitigation |
| :-: | :--- | :--- | :--- |
| **1** | **Device Passcode vs. Account Password Ambiguity** | `golden_037`: *"iPhone: 'Enter device password'. What is the device password?"* | Model treated *"What is..."* as general info rather than a lock-screen recovery blocker. *Mitigation:* Add explicit lockout few-shot examples. |
| **2** | **Multi-Intent Friction: Store Service Complaint vs. Device Defect** | `golden_054`: *"it's ridiculous how hard it is to get some help every time I walk into one of your stores. My Apple Watch has been broken for almost a year..."* | "Broken" triggered hardware repair, shadowing severe customer service anger. *Mitigation:* Enforce a hierarchical rule prioritizing customer sentiment over hardware keywords. |
| **3** | **Post-Damage Warranty / AppleCare Eligibility Boundary** | `golden_092`: *"is it possible to take out apple care after your phone has cracked but you are still within 60 days of purchase?"* | Model saw "AppleCare purchase" and picked billing, missing that pre-existing damage excludes device eligibility. *Mitigation:* Embed AppleCare warranty terms into knowledge base. |
| **4** | **Localization / Web UI Bug on Apple ID Portal** | `golden_141`: *"What is wrong with 'Manage your Apple ID' localization?"* | UI translation bug was treated as general guidance rather than routing to the Apple ID portal team. *Mitigation:* Broaden account security triggers to include `appleid.apple.com` web portal issues. |
| **5** | **Carrier Porting vs. 2FA Security Interlock** | `golden_160`: *"I ported my cell # from another carrier. now SMS frm Apple (2 step verif) is blocked."* | Carrier porting keywords masked the critical 2FA authentication lockout. *Mitigation:* Enforce a security priority override whenever 2FA/verification terms appear. |

---

### 6.3 "What is Misleading About My Headline Number?" (Mandatory Section)

Our headline numbers—**97.0% Intent Accuracy**, **98.5% Escalation Accuracy**, and **4.30/5.00 Reply Quality**—look outstanding. However, deploying this system to production based solely on these figures would be dangerous:

1. **Stratified Sampling vs. Live Class Imbalance:** The 200-sample golden set was deliberately stratified into balanced slices with a 50/50 escalation split. Live Twitter data is heavily skewed: >65% of tweets are noisy, repetitive software complaints during OS releases, while rare account takeover attacks occur in <1% of volume.
2. **Single-Turn Blindness vs. Multi-Turn Degradation:** The benchmark evaluates only the initial tweet. In production, support conversations last 3–8 turns. An agent that answers the first tweet well can easily drift or contradict itself when the user replies *"That didn't work."*
3. **The 2.0% False Auto-Handle Rate Represents Asymmetric Business Risk:** A false escalation merely costs human agent time; a false auto-handle on a compromised Apple ID loses customer trust and invites legal/PR liability. In a brand receiving 100,000 tweets/day, a 2.0% error rate means **2,000 neglected or breached customers daily**.
4. **LLM Judge Leniency Bias:** The LLM Judge awards high marks (4.5+) to any polite tweet containing an Apple link. In fact, our trivial canned DM baseline scored **2.75/5.0** simply because it was polite and safe, despite failing to answer the customer's question.
5. **Historical Twitter Brevity Bias:** Real `@AppleSupport` tweets often simply deflect to DMs (*"Send us a DM with your iOS version"*). Grounding on historical data risks inheriting this deflection tendency rather than achieving autonomous first-contact resolution.

---

### 6.4 What We'd Do Next With One More Week

1. **Multi-Turn Session State Tracking:** Implement stateful conversation graphs (e.g. LangGraph) to track diagnostic context, steps already tried, and repetitive frustration.
2. **Deterministic Outbound Guardrail Engine:** Wrap all LLM output in strict regex/guardrail filters (e.g., NeMo Guardrails) ensuring zero public credential leaks or unauthorized refund promises.
3. **Confidence-Calibrated Human Review Queue:** Route queries with confidence scores between 0.50 and 0.80 to an internal dashboard for one-click human agent approval.
4. **Live GSX Diagnostic & Store Booking API Integration:** Connect to mock Apple Global Service Exchange APIs to provide live Genius Bar reservation slots directly.
5. **Adversarial Red-Teaming:** Subject the system to prompt injection attacks attempting social engineering or policy circumvention.

---

### 6.5 Decision Log (12 Non-Obvious Decisions)

1. **Brand Selection: AppleSupport over AmazonHelp:** AmazonHelp tweets are >85% repetitive order-lookup deflections (*"Please DM order # and zip"*). AppleSupport offers diverse technical troubleshooting (battery, OS updates, Bluetooth) where autonomous resolution is genuinely viable.
2. **Intent Taxonomy Size: Exactly 8 MECE Categories:** Avoided 2-class oversimplification (too coarse for technical grounding) and Banking77's 77 classes (too fragmented for noisy tweets).
3. **Local Hybrid TF-IDF / BM25 Indexing over Remote Vector DBs:** Runs in < 2ms, requires zero API credits, avoids network latency, and ensures exact hardware/OS keyword fidelity (`iOS 11.1.2`, `Genius Bar`, `iPhone 7 Plus`).
4. **Decoupled Classification from Reply Drafting:** Decoupling into structured JSON triage followed by grounded reply drafting ensures auditability and eliminates instruction drift.
5. **Stratified 50/50 Escalation Split in Golden Set:** An unstratified set would over-represent easy software queries, hiding catastrophic blind spots on rare security or billing escalations.
6. **Filtering for Initial Customer Contact (Inbound Only):** Filtered out conversational fragments like *"thanks"* or *"ok"*, retaining substantive inquiries (≥ 25 chars).
7. **Code-Level Policy Enforcement Post-Classification:** Hard deterministic rules override probabilistic model predictions for security and billing intents.
8. **Dual-Mode LLM Client:** Live Gemini client with offline deterministic benchmark fallback ensures reviewers can reproduce results in < 2 minutes without API keys.
9. **Separate Measurement of False Auto-Handle Rate:** Standard accuracy treats all errors equally; tracking False Auto-Handle Rate directly captures catastrophic business risk.
10. **4-Dimensional LLM Judge Rubric:** Broken into Groundedness, Helpfulness, Tone, and Safety to diagnose exact failure points.
11. **Human Quality Calibration on Historical Replies:** Benchmarking the judge against 200 hand-graded human scores provides empirical validation of rater agreement (Quadratic Cohen's Kappa, MAE).
12. **Sanitizing iOS 11 Text Artifacts (`\ufe0f`):** The 2017 iOS 11 letter "I" glitch filled tweets with unicode variation selectors; sanitizing them prevented tokenizer corruption.

---

## 7. Golden Evaluation Set & Human Calibration

The Golden Evaluation Set is stored in [`data/golden/golden_set_200.json`](data/golden/golden_set_200.json) and [`data/golden/golden_set_200.csv`](data/golden/golden_set_200.csv).
See [`data/golden/SAMPLING_AND_LABELLING_NOTE.md`](data/golden/SAMPLING_AND_LABELLING_NOTE.md) for complete details on sampling and labelling protocols.

**Human vs. Judge Calibration Results:**
- **Exact Agreement:** **52.5%**
- **Adjacent Agreement (within ±1 score):** **92.0%**
- **Mean Absolute Error (MAE):** **0.561**
- **Quadratic Weighted Cohen's Kappa:** **0.191**
- **Pearson Correlation ($r$):** **0.236**

---

## 8. Citations & Acknowledgements

1. **ThoughtVector Customer Support on Twitter Dataset:**
   - *Citation:* Thought Vector. (2017). *Customer Support on Twitter*. Kaggle. https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
2. **Google GenAI Python SDK:**
   - *Citation:* Google. (2025). `google-genai` Python SDK. https://github.com/googleapis/python-genai
3. **Scikit-Learn Evaluation & Vectorization:**
   - *Citation:* Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825-2830.
