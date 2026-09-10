# Golden Evaluation Set: Sampling & Labelling Methodology

## 1. Overview
The Golden Evaluation Set contains **200 hand-curated and verified examples** derived from real customer interactions with `@AppleSupport` on Twitter (from the `thoughtvector/customer-support-on-twitter` dataset).

The set serves as the ground truth benchmark for evaluating:
1. **Intent Classification** (8-class taxonomy)
2. **Escalation & Triage Decision** (Binary auto-handle vs. escalate)
3. **Escalation Reasoning** (Stated policy rationale)
4. **Grounded Reply Quality** (Evaluated against human-annotated key resolution points)

---

## 2. Sampling Strategy
To prevent evaluation bias and avoid skew towards high-volume trivial queries, we applied **stratified balanced sampling**:

| Intent Category | Count | Primary Inquiry Types | Escalation Ground Truth |
| :--- | :--- | :--- | :--- |
| `software_update_os` | 30 | iOS 11 letter "I" glitch, freezing, update crashes | Auto-Handle (troubleshooting) |
| `battery_performance` | 25 | Rapid drainage, dying at 20%, overheating | Auto-Handle (diagnostic guidance) |
| `connectivity_peripherals` | 25 | Bluetooth dropouts, Wi-Fi errors, AirPods sync | Auto-Handle (network resets) |
| `general_guidance` | 20 | Settings navigation, feature how-tos, compatibility | Auto-Handle (informational) |
| `account_apple_id_security` | 25 | Locked Apple ID, 2FA recovery, stolen device | **Escalate** (security/PII policy) |
| `billing_subscriptions` | 25 | Unauthorized charges, double billing, refunds | **Escalate** (financial policy) |
| `hardware_repair_service` | 25 | Cracked screens, button damage, Genius Bar visits | **Escalate** (physical service) |
| `escalation_human_complaint` | 25 | Severe customer anger, multiple failed attempts | **Escalate** (dissatisfaction policy) |
| **Total** | **200** | **Balanced across all support dimensions** | **100 Auto-Handle / 100 Escalate** |

### Stratification Rationale
- **50/50 Escalation Balance**: An unstratified sample would over-represent trivial queries (where auto-handling is easy), hiding catastrophic failures on sensitive security or billing complaints. The 50/50 balance rigorously tests both precision and recall.
- **Deduplication & Length Filter**: Filtered for initial customer contact tweets (eliminating multi-turn conversational noise like "thanks!" or "ok"), with length ≥ 25 characters to ensure substantive semantic context.

---

## 3. Labelling Schema & Adjudication Rules
Each example was labelled with:
1. `true_intent`: Assigned to one of 8 mutually exclusive categories based on the customer's primary root-cause issue.
2. `true_should_escalate`:
   - `True` if resolution requires account credentials, financial refunds, physical hardware inspection, or customer is irate.
   - `False` if issue can be resolved safely via verified troubleshooting steps or public documentation links.
3. `true_escalation_reason`: Categorized into one of five policy rules (`ACCOUNT_SECURITY`, `BILLING_DISPUTE`, `HARDWARE_REPAIR`, `FRUSTRATED_CUSTOMER`, or `LOW_CONFIDENCE`), or `AUTO_HANDLE`.
4. `key_resolution_points`: Standard technical criteria expected in a grounded reply (e.g., specific diagnostic questions, recommended settings adjustments, or official Apple portal links).
5. `human_quality_score`: A 1–5 human rating of the original historical Apple tweet, used to benchmark human agreement with our automated LLM judge.
