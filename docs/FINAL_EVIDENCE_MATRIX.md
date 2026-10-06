# 32040 Industry Project — Final Evidence Matrix

## Project

**Title:** Business Context-Aware Insider Threat Detection Using Banking Audit Logs

**Final technical approach:** Role-specific behavioural anomaly detection using Isolation Forest, deterministic context-aware rules, and an LLM-assisted contextual investigation layer.

**Data:** Synthetic banking audit logs only.

**Observation unit:** One employee × one day.

---

# 1. Final Research Question

How can role-specific behavioural anomaly detection, supported by contextual analysis, identify unusual employee activity in synthetic banking audit logs and provide useful evidence for security investigation?

The project evaluates this question using three controlled anomaly-difficulty levels:

1. Obvious anomalies
2. Moderate anomalies
3. Subtle anomalies

The primary experimental question is:

How does the performance of role-specific behavioural anomaly detection change as anomalous behaviour becomes progressively closer to normal role-specific behaviour?

---

# 2. Final System Architecture

Synthetic Banking Audit Logs
        |
        v
User-Day Feature Engineering
        |
        +----------------------+
        |                      |
        v                      v
Role-Specific            Deterministic
Isolation Forest             Rules
        |                      |
        v                      v
ML Prediction            Rule Prediction
        |                      |
        +----------+-----------+
                   |
                   v
          Candidate Selection
            (ML OR Rules)
                   |
                   v
       Structured Context Package
                   |
                   v
       LLM Contextual Assessment
                   |
                   v
          Analyst Investigation
                   |
                   v
          Streamlit Dashboard

The LLM is not treated as an independent anomaly detector and does not infer malicious intent. It provides contextual interpretation and investigation prioritisation for candidates already surfaced by upstream detection.

---

# 3. Synthetic Dataset Evidence

## Training

- 30 employees
- 10 employees per role
- Roles:
  - Teller
  - Customer Service
  - Manager
- 30 working days
- 34,446 raw audit events
- 900 user-day observations
- 300 user-days per role

Purpose:

Train role-specific Isolation Forest models using synthetic normal behaviour.

## Validation

- Independent normal synthetic dataset
- 15 working days
- 450 user-day observations
- 150 per role

Purpose:

Calibrate frozen role-specific anomaly thresholds independently from the test experiments.

Threshold method:

95th percentile of unseen-normal anomaly scores.

Project anomaly score:

anomaly_score = -IsolationForest.score_samples(X)

Higher score therefore represents more unusual behaviour within this project implementation.

## Test

Each experiment contains:

- 300 user-days
- 270 normal user-days
- 30 deliberately injected anomalous user-days
- Synthetic experimental anomaly prevalence = 10%

The 10% prevalence is an experimental design choice and must not be represented as real banking anomaly prevalence.

---

# 4. ML Features

The frozen Isolation Forest feature set is:

1. event_count
2. transaction_count
3. customer_record_access_count
4. download_event_count
5. download_total
6. records_accessed_total
7. vip_access_count
8. first_activity_hour
9. last_activity_hour
10. activity_duration_hours

Contextual fields such as failed_login_count, unknown_ip_count, and unique_ip_count are not part of the frozen Isolation Forest feature set.

No StandardScaler is used in the final implementation.

One Isolation Forest model is trained independently for each role.

---

# 5. Frozen ML Thresholds

Customer Service:
0.5521464960734026

Manager:
0.5379024781433652

Teller:
0.5341784647040614

These thresholds were calibrated using independent normal validation data and were frozen before test evaluation.

They were not recalibrated after observing experiment results.

---

# 6. Experimental Design

The same 30 anomaly identities and scenario assignments are used across all three experiments.

Only anomaly strength changes.

This provides a controlled comparison between:

Experiment 1 — Obvious

Experiment 2 — Moderate

Experiment 3 — Subtle

Three anomaly scenario families are used:

1. Unusual Export Behaviour
2. Unusual Customer Record Access
3. Temporal + Combined Behaviour

Per experiment:

- 9 unusual export cases
- 9 unusual customer-record-access cases
- 12 temporal + combined cases

Anomalies are injected into raw synthetic audit events before the standard feature-engineering pipeline.

Ground-truth labels are stored separately and are not available during blind detector prediction.

---

# 7. Isolation Forest Results

## Experiment 1 — Obvious

TP = 14
TN = 257
FP = 13
FN = 16

Accuracy = 90.33%
Precision = 51.85%
Recall = 46.67%
F1 = 49.12%
FPR = 4.81%

## Experiment 2 — Moderate

TP = 14
TN = 257
FP = 13
FN = 16

Accuracy = 90.33%
Precision = 51.85%
Recall = 46.67%
F1 = 49.12%
FPR = 4.81%

## Experiment 3 — Subtle

TP = 8
TN = 257
FP = 13
FN = 22

Accuracy = 88.33%
Precision = 38.10%
Recall = 26.67%
F1 = 31.37%
FPR = 4.81%

---

# 8. ML Scenario-Level Findings

Experiment 1 / Experiment 2:

Temporal + Combined:
12 / 12 detected

Unusual Export:
1 / 9 detected

Unusual Customer Record Access:
1 / 9 detected

Experiment 3:

Temporal + Combined:
7 / 12 detected

Unusual Export:
0 / 9 detected

Unusual Customer Record Access:
1 / 9 detected

Interpretation:

The Isolation Forest was substantially more responsive to combined multivariate behavioural deviations than isolated single-behaviour deviations.

Obvious and moderate experiments produced identical classification outcomes.

Detection performance decreased materially for subtle anomalies.

Do not claim that performance decreased uniformly across all three difficulty levels.

---

# 9. Deterministic Rule Baseline

Final rules:

1. After-hours activity
2. Large export volume
3. High customer-record access

Role-specific rule thresholds were calibrated from normal validation behaviour.

A user-day is flagged when at least one rule triggers.

## Experiments 1 and 2

TP = 30
TN = 228
FP = 42
FN = 0

Accuracy = 86.00%
Precision = 41.67%
Recall = 100.00%
F1 = 58.82%
FPR = 15.56%

## Experiment 3

TP = 26
TN = 228
FP = 42
FN = 4

Accuracy = 84.67%
Precision = 38.24%
Recall = 86.67%
F1 = 53.06%
FPR = 15.56%

Interpretation:

Rules provide substantially higher recall than the Isolation Forest in these synthetic experiments, but at the cost of a substantially higher false-positive rate.

Rules are not universally "better" than ML.

---

# 10. Detector Overlap

Experiments 1 and 2 — anomaly cases:

Both ML and rules: 14
Rules only: 16
ML only: 0
Missed by both: 0

Experiment 3 — anomaly cases:

Both: 8
Rules only: 18
ML only: 0
Missed by both: 4

Normal user-days, all experiments:

Both detectors: 9 false positives
ML only: 4 false positives
Rules only: 33 false positives
Neither: 224

Important implication:

In these specific synthetic scenarios, the ML detector did not identify an injected anomaly that was not also identified by the rule baseline.

This is an experimental finding and must not be generalised beyond the tested scenarios.

---

# 11. LLM Input Design

Candidate selection:

ML anomaly OR rule anomaly.

Candidate counts:

Experiment 1 = 76
Experiment 2 = 76
Experiment 3 = 72

Total LLM assessments = 224.

LLM input contains:

- employee role
- role/business context
- aggregated user-day behavioural features
- ML anomaly score
- frozen ML threshold
- ML prediction
- deterministic rule evidence
- number of triggered rules

LLM input excludes:

- true_label
- case_id
- injected scenario
- anomaly difficulty
- experiment difficulty
- raw event timeline

This separation prevents direct ground-truth leakage into the contextual assessment.

---

# 12. LLM Configuration

Frozen model:

gpt-5-nano-2025-08-07

Frozen prompt:

Version 2

Output schema:

- risk_level
- investigation_priority
- summary
- evidence
- role_context
- uncertainty
- recommended_action

Risk mapping:

Low -> Routine

Medium -> Review

High -> Escalate

The prompt explicitly prevents malicious-intent or confirmed-compromise attribution.

---

# 13. LLM Risk Distribution

Experiment 1:

Low = 0
Medium = 65
High = 11

Experiment 2:

Low = 0
Medium = 70
High = 6

Experiment 3:

Low = 0
Medium = 70
High = 2

Important result:

No candidate received a Low assessment.

Therefore, under the predefined Low versus Medium/High binary mapping, the LLM did not reduce the investigation-candidate workload.

---

# 14. High-Risk Prioritisation

Experiment 1:

11 High assessments
10 injected anomalies
1 normal candidate

High-risk precision = 90.91%

Experiment 2:

6 High assessments
6 injected anomalies

High-risk precision = 100%

Experiment 3:

2 High assessments
2 injected anomalies

High-risk precision = 100%

All injected anomaly cases assigned High risk were from the Temporal + Combined scenario family.

Interpretation:

The High-risk tier concentrated strongly on combined behavioural deviations.

Because the High sample is small and the dataset is synthetic, this must not be presented as evidence of real-world calibrated risk prediction.

---

# 15. LLM-Assisted End-to-End Binary Results

## Experiment 1

TP = 30
TN = 224
FP = 46
FN = 0

Accuracy = 84.67%
Precision = 39.47%
Recall = 100%
F1 = 56.60%
FPR = 17.04%

## Experiment 2

Same as Experiment 1.

## Experiment 3

TP = 26
TN = 224
FP = 46
FN = 4

Accuracy = 83.33%
Precision = 36.11%
Recall = 86.67%
F1 = 50.98%
FPR = 17.04%

Important interpretation:

The LLM-assisted pipeline did not improve binary anomaly-detection metrics compared with the rule-only baseline.

Because no candidate was classified Low, the LLM did not remove false-positive candidates.

The candidate union also contains four normal cases identified only by ML, which increases false positives compared with rules alone.

The value of the LLM in this experiment is therefore contextual interpretation and prioritisation, rather than improved binary anomaly detection.

---

# 16. LLM Qualitative Evaluation

Manual review sample:

34 assessments

Sampling:

- all 19 High-risk assessments
- 15 Medium-risk assessments selected using a fixed seed

Results:

Grounding:
33 / 34 = 97.1%

Role awareness:
33 / 34 = 97.1%

Unsupported-claim control:
33 / 34 = 97.1%

Uncertainty handling:
34 / 34 = 100%

Completely clean across all four criteria:
32 / 34 = 94.1%

Two material issues were identified:

1. One assessment incorrectly described download_total as the number of export events.

2. One assessment stated that substantial export activity was not unusual for a Teller, conflicting with the supplied Teller role context.

Interpretation:

The LLM was generally grounded and role-aware but was not error-free.

Human oversight remains necessary.

---

# 17. Final Comparative Results

| Difficulty | Method | Accuracy | Precision | Recall | F1 | FPR |
|---|---|---:|---:|---:|---:|---:|
| Obvious | Isolation Forest | 90.33% | 51.85% | 46.67% | 49.12% | 4.81% |
| Obvious | Rules | 86.00% | 41.67% | 100.00% | 58.82% | 15.56% |
| Obvious | LLM-assisted | 84.67% | 39.47% | 100.00% | 56.60% | 17.04% |
| Moderate | Isolation Forest | 90.33% | 51.85% | 46.67% | 49.12% | 4.81% |
| Moderate | Rules | 86.00% | 41.67% | 100.00% | 58.82% | 15.56% |
| Moderate | LLM-assisted | 84.67% | 39.47% | 100.00% | 56.60% | 17.04% |
| Subtle | Isolation Forest | 88.33% | 38.10% | 26.67% | 31.37% | 4.81% |
| Subtle | Rules | 84.67% | 38.24% | 86.67% | 53.06% | 15.56% |
| Subtle | LLM-assisted | 83.33% | 36.11% | 86.67% | 50.98% | 17.04% |

---

# 18. Final Technical Findings

1. Role-specific Isolation Forest provided the most selective detector, maintaining a 4.81% false-positive rate, but had relatively low sensitivity to the injected anomaly scenarios.

2. Deterministic rules provided substantially higher recall but generated substantially more false positives.

3. Detection difficulty did not degrade uniformly. Obvious and moderate experiments produced identical ML classification outcomes, while subtle anomalies materially reduced recall.

4. Combined temporal and behavioural deviations were substantially easier for the Isolation Forest to identify than isolated export or customer-record-access deviations.

5. The LLM did not improve binary anomaly-detection performance under the predefined decision mapping.

6. The LLM nevertheless showed useful contextual prioritisation: High-risk assessments were strongly concentrated on injected combined anomalies.

7. Manual qualitative evaluation found generally strong grounding, role awareness, and uncertainty communication, but identified two material errors.

8. The evidence supports a human-in-the-loop design rather than autonomous threat attribution.

---

# 19. Final Answer to Research Question

Role-specific behavioural anomaly detection can identify unusual employee-day behaviour by learning separate normal behavioural patterns for banking roles and comparing unseen behaviour against role-specific anomaly thresholds.

In the synthetic experiments, the Isolation Forest approach provided relatively selective anomaly detection but was less sensitive to isolated and subtle deviations, while deterministic rules achieved higher recall at the cost of substantially more false positives.

The contextual LLM did not improve binary detection performance, but provided generally grounded, role-aware explanations and effectively prioritised a smaller subset of stronger combined anomalies as High risk.

The results therefore support a layered human-in-the-loop design in which ML and deterministic rules surface anomaly candidates, while the LLM provides contextual evidence and investigation prioritisation rather than autonomous threat attribution.

---

# 20. Project Evolution from Proposal

The original proposal focused primarily on:

- context-aware deterministic rules
- an LLM risk-assessment layer
- Low / Medium / High classification
- comparison with a rule-only baseline

During implementation and supervisory refinement, the project evolved to include role-specific behavioural anomaly detection using Isolation Forest.

The final system therefore evaluates three complementary components:

1. Role-specific behavioural ML
2. Role-specific deterministic rules
3. LLM contextual assessment of surfaced candidates

This evolution must be stated transparently in the final report.

The project must not claim that Isolation Forest was part of the original proposal.

---

# 21. Scope Refinement

The original proposal included four illustrative rules:

- repeated failed login attempts
- after-hours access
- repeated access to the same customer record
- large-volume data export

The final evaluated rule baseline uses:

- after-hours activity
- large export volume
- high customer-record access

Repeated same-customer access was not implemented because the final synthetic user-day model does not include customer identity.

A customer_id field was deliberately not added solely to preserve an originally proposed rule.

Customer sensitivity remains represented through synthetic account categories and VIP-access behaviour.

This is a scope refinement and must be documented transparently.

---

# 22. Major Limitations

1. All data are synthetic.

2. Results do not establish performance on a real banking environment.

3. Synthetic experimental anomaly prevalence of 10% is not representative of real-world prevalence.

4. Synthetic role behaviour is based on project assumptions, not empirically estimated production banking statistics.

5. Raw anomaly injection prioritises aggregate behavioural targets and is not a complete simulation of realistic attacker workflows.

6. Isolation Forest sensitivity was weak for isolated single-feature anomalies.

7. The rule baseline produced a relatively high false-positive rate.

8. The LLM can only analyse candidates surfaced by upstream detection.

9. Four subtle anomalies in Experiment 3 were missed by both upstream detectors and therefore never reached the LLM.

10. The LLM did not reduce candidate workload under the frozen binary decision mapping.

11. LLM qualitative assessment demonstrated occasional evidence and role-context errors.

12. The LLM qualitative audit covers a structured sample of 34 of 224 outputs rather than every assessment.

13. The system must not infer malicious intent solely from behavioural anomaly detection.

---

# 23. Reproducibility Evidence

Stable end-to-end pipeline:

src/pipeline/detection_pipeline.py

Runtime CLI:

src/pipeline/run_pipeline.py

Frozen experiment verification:

src/pipeline/verify_frozen_experiments.py

Verification results:

Experiment 1:
ML = 27
Rules = 72
Candidates = 76

Experiment 2:
ML = 27
Rules = 72
Candidates = 76

Experiment 3:
ML = 21
Rules = 68
Candidates = 72

Frozen ML identity:
PASS

Frozen rule identity:
PASS

No retraining:
PASS

No threshold recalibration:
PASS

No ground-truth use during detection:
PASS

No OpenAI API call during reproduction:
PASS

---

# 24. Prototype Evidence

Implemented prototype components:

- synthetic audit-log generation
- audit-log processing
- user-day feature engineering
- role-specific Isolation Forest training
- independent validation
- frozen threshold calibration
- blind experiment scoring
- deterministic rule baseline
- ground-truth evaluation
- LLM context-package generation
- LLM contextual analysis
- quantitative LLM evaluation
- qualitative LLM evaluation
- reproducible end-to-end detection pipeline
- Streamlit analyst dashboard

The Streamlit dashboard provides:

- experiment selection
- detection summary
- role-level summary
- investigation candidate table
- employee-day investigation view
- ML evidence
- deterministic-rule evidence
- behavioural features
- frozen contextual LLM assessment
- human-in-the-loop disclaimer

---

# 25. Claims We CAN Make

We can state:

- The prototype successfully implements role-specific behavioural anomaly detection over synthetic banking audit logs.

- Isolation Forest had lower false-positive rates than the deterministic rule baseline in the tested synthetic experiments.

- Rules had higher recall than Isolation Forest in the tested scenarios.

- Combined behavioural anomalies were more detectable by the Isolation Forest than isolated export and record-access deviations.

- Subtle anomalies reduced Isolation Forest recall.

- The LLM did not improve binary detection metrics under the frozen evaluation mapping.

- High-risk LLM assessments were strongly concentrated on injected combined anomalies.

- The LLM generally produced grounded and role-aware contextual assessments in the structured manual review.

- Human oversight remains necessary.

---

# 26. Claims We MUST NOT Make

Do not claim:

- anomaly = malicious insider

- anomaly = cyberattack

- the system proves compromise

- the system predicts employee intent

- the system prevents ransomware

- the system has been validated on a real bank

- synthetic thresholds represent actual banking policy

- 10% is a realistic insider-threat prevalence

- the 95th-percentile threshold guarantees a 5% false-positive rate

- the LLM improved anomaly-detection accuracy

- the LLM reduced false positives

- the LLM recovered anomalies missed by upstream detection

- Experiment 1, 2, and 3 showed a perfectly monotonic performance decline

- the final Isolation Forest design was part of the original proposal

- repeated access to the same customer was implemented

- the LLM was error-free

---

# 27. Report Evidence Mapping

## Abstract

Evidence:
- final architecture
- three experiments
- final comparative metrics
- principal findings
- limitations

## 1 Introduction

Evidence:
- banking audit-monitoring problem
- role-dependent behaviour
- project motivation
- human-in-the-loop objective

## 2 Background

Evidence:
- audit logs
- insider-threat/anomaly-monitoring context
- business context
- deterministic detection
- behavioural anomaly detection
- Isolation Forest
- LLM-assisted security analysis

External academic references are required.

## 3 Project Goal and Question

Evidence:
- final research question
- experimental question
- objectives
- project evolution from proposal

## 4 Related Works

Evidence:
- external literature

Do not derive literature claims solely from project code.

## 5 Requirements Gathering

Evidence:
- proposal requirements
- supervisor evaluation feedback
- banking role context
- synthetic-data constraint
- human-in-the-loop requirement
- scope refinements

## 6 Proposed Solution

### 6.1 Methodology

Evidence:
- design-build-evaluate
- training/validation/test separation
- controlled experiments
- independent ground truth
- evaluation metrics
- LLM evaluation methodology

### 6.2 Solution Design

Evidence:
- final architecture
- role-specific modelling
- candidate selection
- LLM input design
- no-ground-truth leakage

### 6.3 Solution Implementation

Evidence:
- source-code modules
- configuration files
- models
- pipeline
- dashboard

### 6.4 Solution Test

Evidence:
- Exp1/2/3 results
- ML metrics
- rules metrics
- LLM quantitative results
- qualitative audit
- reproduction test

## 7 Solution Discussion

Evidence:
- detector trade-offs
- scenario sensitivity
- Exp1 vs Exp2 identical outcome
- subtle anomaly degradation
- LLM prioritisation
- failure to reduce false positives
- human-in-the-loop interpretation

## 8 Limitations of Project

Evidence:
- limitations listed in Section 22 of this matrix

## 9 Future Development / Work

Potential future work:
- real-world or higher-fidelity synthetic validation
- richer customer/entity modelling
- coherent event-session anomaly generation
- alternative anomaly-detection models
- temporal/sequential models
- better rule-combination strategies
- calibrated LLM triage
- broader qualitative assessment
- integration with SIEM workflows

These should be presented as future work, not completed work.

## 10 Conclusion

Evidence:
- final research-question answer
- final comparative findings
- limitations
- human-in-the-loop conclusion

## 11 Acknowledgement

Use actual supervisor/project acknowledgements only.

## 12 References

Only verifiable references actually used in the report.