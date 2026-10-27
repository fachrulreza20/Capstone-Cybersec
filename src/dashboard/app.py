"""UTS 32040 Industry Project — final-report-aligned Streamlit dashboard.

Presentation-only update: fixed detector models/thresholds, synthetic test data,
original LLM assessments, and robustness result CSVs are never modified here.
"""

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
from src.pipeline.detection_pipeline import run_raw_pipeline

RAW_DIR = PROJECT_ROOT / "data" / "raw"
LLM_RESULTS_DIR = PROJECT_ROOT / "outputs" / "llm_results"
ROBUSTNESS_DIR = PROJECT_ROOT / "outputs" / "robustness"
PERFORMANCE_FILE = ROBUSTNESS_DIR / "final_performance_table.csv"
LLM_CONTROL_FILE = ROBUSTNESS_DIR / "llm_control_summary.csv"

EXPERIMENTS = {
    "Experiment 1 — Obvious": {
        "raw_file": RAW_DIR / "experiment_1_test.csv",
        "llm_file": LLM_RESULTS_DIR / "experiment_1_llm_results.json",
        "difficulty": "Obvious",
        "ml_expected": 27,
        "rules_expected": 72,
        "candidates_expected": 76,
    },
    "Experiment 2 — Moderate": {
        "raw_file": RAW_DIR / "experiment_2_test.csv",
        "llm_file": LLM_RESULTS_DIR / "experiment_2_llm_results.json",
        "difficulty": "Moderate",
        "ml_expected": 27,
        "rules_expected": 72,
        "candidates_expected": 76,
    },
    "Experiment 3 — Subtle": {
        "raw_file": RAW_DIR / "experiment_3_test.csv",
        "llm_file": LLM_RESULTS_DIR / "experiment_3_llm_results.json",
        "difficulty": "Subtle",
        "ml_expected": 21,
        "rules_expected": 68,
        "candidates_expected": 72,
    },
}

CONDITIONS = ["control", "obvious", "moderate", "subtle"]
CONDITION_NAMES = {
    "control": "Control (0% injected)",
    "obvious": "Obvious (10% injected)",
    "moderate": "Moderate (10% injected)",
    "subtle": "Subtle (10% injected)",
}
DETECTOR_NAMES = {"ml": "Isolation Forest", "rules": "Deterministic rules"}

st.set_page_config(
    page_title="Banking Behavioural Anomaly Detection | UTS 32040",
    page_icon="🏦",
    layout="wide",
)


# ------------------------------------------------------------------
# Frozen source loading / checks (no training, test-label access or API)
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_performance(path_string):
    df = pd.read_csv(path_string)
    required = {"condition", "detector", "metric", "mean", "ci95_lower", "ci95_upper"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Performance CSV missing columns: {sorted(missing)}")
    if len(df) != 36:
        raise ValueError(f"Expected 36 final performance metric rows; found {len(df)}")
    return df


@st.cache_data(show_spinner=False)
def load_control_summary(path_string):
    df = pd.read_csv(path_string)
    required = {"control_group", "risk_level", "count", "percentage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"LLM diagnostic CSV missing columns: {sorted(missing)}")
    return df


@st.cache_data(show_spinner=False)
def load_initial_experiment(experiment_key):
    config = EXPERIMENTS[experiment_key]
    raw = pd.read_csv(config["raw_file"], parse_dates=["timestamp"])
    result = run_raw_pipeline(raw.copy())
    return raw, result["features"], result["combined"].copy()


@st.cache_data(show_spinner=False)
def load_llm_results(path_string):
    path = Path(path_string)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def normalize_date(value):
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def candidate_key(user_id, date, role):
    return (str(user_id).strip(), normalize_date(date), str(role).strip())


def extract_llm_index(data):
    """Find stored assessments by (user_id, date, role); don't call an API."""
    index = {}

    def walk(obj):
        if isinstance(obj, dict):
            assessment = None
            if isinstance(obj.get("assessment"), dict):
                assessment = obj["assessment"]
            elif "risk_level" in obj and "investigation_priority" in obj:
                assessment = obj
            # Frozen batch records store inputs under "candidate_input".
            # Keep compatibility with older records using "candidate".
            candidate = (
                obj.get("candidate_input")
                if isinstance(obj.get("candidate_input"), dict)
                else obj.get("candidate")
                if isinstance(obj.get("candidate"), dict)
                else obj
            )
            context = candidate.get("employee_context") if isinstance(candidate, dict) else None
            if assessment is not None and isinstance(context, dict):
                if all(context.get(x) is not None for x in ["user_id", "date", "role"]):
                    key = candidate_key(context["user_id"], context["date"], context["role"])
                    index[key] = assessment
            for child in obj.values():
                walk(child)
        elif isinstance(obj, list):
            for child in obj:
                walk(child)

    walk(data)
    return index


def fmt_ci(row):
    return f"{row['mean'] * 100:.2f}% (95% CI {row['ci95_lower'] * 100:.2f}–{row['ci95_upper'] * 100:.2f}%)"


def metric_row(performance, condition, detector, metric):
    match = performance[
        (performance["condition"] == condition)
        & (performance["detector"] == detector)
        & (performance["metric"] == metric)
    ]
    if len(match) != 1:
        raise ValueError(f"Expected one row for {condition}/{detector}/{metric}, found {len(match)}")
    return match.iloc[0]


def render_frozen_llm(assessment):
    if not assessment:
        st.info("No stored LLM assessment was found for this candidate.")
        return
    col1, col2 = st.columns(2)
    col1.metric("LLM Contextual Risk Assessment", assessment.get("risk_level", "Unavailable"))
    col2.metric("Suggested Investigation Priority", assessment.get("investigation_priority", "Unavailable"))
    st.caption("Risk is a contextual triage rating, not a ground-truth threat label or proof of malicious intent.")
    st.markdown("#### Summary")
    st.write(assessment.get("summary", "Not available."))
    st.markdown("#### Evidence")
    evidence = assessment.get("evidence", [])
    if evidence:
        for item in evidence:
            st.markdown(f"- {item}")
    else:
        st.write("No supporting evidence text available.")
    st.markdown("#### Role Context")
    st.write(assessment.get("role_context", "Not available."))
    st.markdown("#### Uncertainty")
    st.write(assessment.get("uncertainty", "Not available."))
    st.markdown("#### Recommended Action")
    st.write(assessment.get("recommended_action", "Not available."))


# ------------------------------------------------------------------
# Main page: identical frozen evaluation narrative to the final report
# ------------------------------------------------------------------
st.title("Banking Behavioural Anomaly Detection")
st.caption(
    "UTS 32040 Industry Project — Fachrul Reza & Christopher Gustaviano | "
    "Final-report-aligned research results and original case demonstration"
)
st.warning(
    "Synthetic banking audit logs only. A detector flag means unusual behaviour, "
    "not malicious intent or confirmed compromise. Human investigation is required. "
    "This is a research demonstration, not a production monitoring system."
)

st.sidebar.header("Initial Experiment Demo")
selected_experiment = st.sidebar.selectbox("Controlled experiment", list(EXPERIMENTS.keys()))
st.sidebar.caption(
    "The three choices are original single-run case studies (300 user-days each), "
    "not the final 50-replication performance summaries."
)
st.sidebar.markdown("**Candidate selection:** Isolation Forest OR deterministic rules")
st.sidebar.markdown("**LLM:** Frozen stored assessments; no live API calls")

final_tab, demo_tab, llm_tab, method_tab = st.tabs(
    ["Final Replicated Results", "Initial Case Study & Candidate Demo", "LLM Normal-Control Diagnostic", "Methodology & Limits"]
)

with final_tab:
    st.header("Final Replicated Robustness Evaluation")
    st.caption("Final report: Section 6.4.2–6.4.3, Table 8–9 and Appendix B")
    try:
        performance = load_performance(str(PERFORMANCE_FILE))
        st.markdown(
            "Four conditions were independently replicated 50 times each: "
            "a normal-only control and three conditions with 10% injected anomalies. "
            "One fixed set of role-specific models, ML thresholds and rule thresholds "
            "was evaluated without retraining or recalibration."
        )
        k1, k2, k3 = st.columns(3)
        k1.metric("Independent test datasets", "200")
        k2.metric("Replications per condition", "50")
        k3.metric("Synthetic employee-days", "60,000")

        st.subheader("Main findings")
        m1, m2, m3 = st.columns(3)
        m1.metric("ML control FPR", f"{metric_row(performance, 'control', 'ml', 'fpr')['mean'] * 100:.2f}%")
        m2.metric("Rules control FPR", f"{metric_row(performance, 'control', 'rules', 'fpr')['mean'] * 100:.2f}%")
        m3.metric("ML subtle recall", f"{metric_row(performance, 'subtle', 'ml', 'recall')['mean'] * 100:.2f}%")

        st.subheader("Mean recall — injected anomaly conditions")
        recall_rows = []
        for condition in ["obvious", "moderate", "subtle"]:
            recall_rows.append({
                "Condition": condition.capitalize(),
                "Isolation Forest": 100 * metric_row(performance, condition, "ml", "recall")["mean"],
                "Deterministic rules": 100 * metric_row(performance, condition, "rules", "recall")["mean"],
            })
        recall_chart = pd.DataFrame(recall_rows).set_index("Condition")
        st.bar_chart(recall_chart, y_label="Mean recall (%)")
        st.caption("Bars show means. Exact two-sided 95% confidence intervals are in the table below.")

        st.subheader("Mean false-positive rate — all conditions")
        fpr_rows = []
        for condition in CONDITIONS:
            fpr_rows.append({
                "Condition": condition.capitalize(),
                "Isolation Forest": 100 * metric_row(performance, condition, "ml", "fpr")["mean"],
                "Deterministic rules": 100 * metric_row(performance, condition, "rules", "fpr")["mean"],
            })
        fpr_chart = pd.DataFrame(fpr_rows).set_index("Condition")
        st.bar_chart(fpr_chart, y_label="Mean false-positive rate (%)")
        st.caption("Control contains zero injected anomalies; anomaly recall and precision are not defined for Control.")

        st.subheader("Full replicated results — mean (95% CI)")
        full_rows = []
        for condition in CONDITIONS:
            for detector in ["ml", "rules"]:
                row = {"Condition": CONDITION_NAMES[condition], "Detector": DETECTOR_NAMES[detector]}
                for metric, label in [
                    ("accuracy", "Accuracy"),
                    ("precision", "Precision"),
                    ("recall", "Recall"),
                    ("f1", "F1"),
                    ("fpr", "FPR"),
                ]:
                    row[label] = "N/A" if condition == "control" and metric in {"precision", "recall", "f1"} else fmt_ci(metric_row(performance, condition, detector, metric))
                full_rows.append(row)
        st.dataframe(pd.DataFrame(full_rows), use_container_width=True, hide_index=True)
        with st.expander("Control specificity and metric interpretation"):
            control_specificity = pd.DataFrame([
                {"Detector": DETECTOR_NAMES[detector], "Specificity (mean and 95% CI)": fmt_ci(metric_row(performance, "control", detector, "specificity"))}
                for detector in ["ml", "rules"]
            ])
            st.dataframe(control_specificity, hide_index=True, use_container_width=True)
            st.write(
                "Each figure above is the mean across 50 run-level metrics. The 95% confidence "
                "interval is a two-sided Student's t interval with 49 degrees of freedom. "
                "No formal significance test or paired condition comparison was performed."
            )
        st.info(
            "Interpretation: Isolation Forest is more selective (FPR approximately 5%) "
            "but missed many injected anomalies, especially subtle cases. Rules have higher "
            "recall but FPR approximately 16%. The tested injection scenarios directly modify "
            "features checked by those rules. This is a synthetic-study trade-off, not proof "
            "that either approach is generally superior."
        )
        with PERFORMANCE_FILE.open("rb") as source:
            st.download_button(
                "Download exact final performance table (CSV)",
                data=source.read(),
                file_name="final_performance_table.csv",
                mime="text/csv",
            )
    except Exception as exc:
        st.error(f"Cannot load the final replicated results: {exc}")
        st.caption("Expected file: outputs/robustness/final_performance_table.csv")

with demo_tab:
    st.header("Initial Controlled Experiment — Individual Case Demonstration")
    st.caption("Final report: Sections 6.4.4–6.4.6 and 6.4.9, Table 10–18, Figures 8–9")
    st.info(
        "These counts are from one original fixed experiment, NOT averages over 50 replications. "
        "Each initial dataset includes 300 user-days. The LLM responses below are stored "
        "original assessments."
    )
    st.markdown(f"**Selected case study:** {selected_experiment}")
    try:
        with st.spinner("Rebuilding original case-study predictions from synthetic raw logs..."):
            raw_df, features, combined = load_initial_experiment(selected_experiment)
        llm_index = extract_llm_index(
            load_llm_results(str(EXPERIMENTS[selected_experiment]["llm_file"]))
        )
    except Exception as exc:
        st.error(f"Initial case-study pipeline failed: {exc}")
        st.stop()

    ml_count = int(combined["prediction"].sum())
    rule_count = int(combined["rule_prediction"].sum())
    candidate_count = int(combined["candidate"].sum())
    st.subheader("Detection Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Raw Events", f"{len(raw_df):,}")
    c2.metric("User-Days", f"{len(features):,}")
    c3.metric("ML-Flagged Days", ml_count)
    c4.metric("Rule-Flagged Days", rule_count)
    c5.metric("Investigation Candidates", candidate_count)
    expected = EXPERIMENTS[selected_experiment]
    if (len(raw_df), len(features), ml_count, rule_count, candidate_count) != (
        11524, 300, expected["ml_expected"], expected["rules_expected"], expected["candidates_expected"]
    ):
        st.warning("Counts differ from frozen Table 18 of the submitted report; verify the source files.")

    st.subheader("Role-Level Detection Summary")
    role_summary = (
        combined.groupby("role")
        .agg(
            user_days=("user_id", "size"),
            ml_flagged=("prediction", "sum"),
            rule_flagged=("rule_prediction", "sum"),
            candidates=("candidate", "sum"),
        )
        .reset_index()
        .rename(columns={"role": "Role", "user_days": "User-Days", "ml_flagged": "ML-Flagged", "rule_flagged": "Rule-Flagged", "candidates": "Candidates"})
    )
    st.dataframe(role_summary, use_container_width=True, hide_index=True)

    st.subheader("Investigation Candidates")
    candidates = combined[combined["candidate"] == 1].copy()
    candidates["ML Status"] = candidates["prediction"].map({1: "Flagged", 0: "Not flagged"})
    candidates["Rule Status"] = candidates["rule_prediction"].map({1: "Flagged", 0: "Not flagged"})
    candidate_columns = ["user_id", "date", "role", "ML Status", "anomaly_score", "threshold", "Rule Status", "rules_triggered"]
    candidate_display = candidates[candidate_columns].rename(columns={
        "user_id": "User", "date": "Date", "role": "Role", "anomaly_score": "ML Anomaly Score",
        "threshold": "ML Threshold", "rules_triggered": "Rules Triggered",
    })
    st.dataframe(candidate_display, use_container_width=True, hide_index=True)
    if candidates.empty:
        st.info("No candidates in the selected initial experiment.")
    else:
        st.subheader("Candidate Investigation")
        candidate_lookup = {}
        for _, candidate in candidates.iterrows():
            key = candidate_key(candidate["user_id"], candidate["date"], candidate["role"])
            label = f"{key[0]} | {key[1]} | {key[2]}"
            candidate_lookup[label] = key
        chosen_label = st.selectbox("Select employee-day", list(candidate_lookup))
        chosen_key = candidate_lookup[chosen_label]
        selected = candidates[
            (candidates["user_id"].astype(str).str.strip() == chosen_key[0])
            & (pd.to_datetime(candidates["date"]).dt.strftime("%Y-%m-%d") == chosen_key[1])
            & (candidates["role"].astype(str).str.strip() == chosen_key[2])
        ]
        if len(selected) != 1:
            st.error("Selected candidate cannot be mapped to exactly one employee-day.")
        else:
            row = selected.iloc[0]
            st.markdown("### Employee-Day Context")
            e1, e2, e3 = st.columns(3)
            e1.metric("Employee", str(row["user_id"]))
            e2.metric("Role", str(row["role"]))
            e3.metric("Date", normalize_date(row["date"]))

            st.markdown("### ML Evidence")
            ml1, ml2, ml3 = st.columns(3)
            ml1.metric("ML Prediction", "Flagged" if int(row["prediction"]) else "Not flagged")
            ml2.metric("Anomaly Score", f"{row['anomaly_score']:.6f}")
            ml3.metric("Frozen Role Threshold", f"{row['threshold']:.6f}")
            st.caption(
                "Higher anomaly score means more unusual behaviour. The role-specific "
                "threshold was calibrated on separate normal validation data and not changed during testing."
            )

            st.markdown("### Deterministic Rule Evidence")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Late Activity", "Yes" if int(row["rule_after_hours"]) else "No")
            r2.metric("Large Export", "Yes" if int(row["rule_large_export"]) else "No")
            r3.metric("High Record Access", "Yes" if int(row["rule_high_record_access"]) else "No")
            r4.metric("Rules Triggered", int(row["rules_triggered"]))

            st.markdown("### Behavioural Summary")
            behaviour_columns = [
                "event_count", "transaction_count", "customer_record_access_count",
                "download_event_count", "download_total", "records_accessed_total",
                "vip_access_count", "failed_login_count", "unknown_ip_count",
                "unique_ip_count", "first_activity_hour", "last_activity_hour",
                "activity_duration_hours",
            ]
            available = [name for name in behaviour_columns if name in row.index]
            st.dataframe(pd.DataFrame({"Feature": available, "Value": [row[name] for name in available]}), use_container_width=True, hide_index=True)

            st.markdown("### Contextual LLM Assessment (frozen original output)")
            render_frozen_llm(llm_index.get(chosen_key))

with llm_tab:
    st.header("LLM Normal-Control Diagnostic")
    st.caption("Final report: Sections 6.4.7 and 7.5, Tables 15–16")
    st.write(
        "The original candidate-only assessment rated all 224 candidates Medium or High. "
        "To investigate why no Low ratings appeared, the frozen model and prompt v2 "
        "were tested on 60 deliberately sampled **ground-truth-normal** employee-days "
        "from the 50 normal-only replications. Each group had 10 employees-days per role."
    )
    try:
        control = load_control_summary(str(LLM_CONTROL_FILE))
        counts = control.pivot_table(index="control_group", columns="risk_level", values="count", aggfunc="sum", fill_value=0)
        counts = counts.reindex(index=["flagged_normal", "unflagged_normal"], columns=["Low", "Medium", "High"], fill_value=0)
        counts.index = ["Flagged normal", "Unflagged normal"]
        if counts.to_numpy().sum() != 60:
            st.warning("Diagnostic case total differs from 60; check the stored output.")
        col1, col2 = st.columns(2)
        col1.metric("Flagged-normal cases assigned Low", f"{int(counts.loc['Flagged normal', 'Low'])} / 30")
        col2.metric("Unflagged-normal cases assigned Low", f"{int(counts.loc['Unflagged normal', 'Low'])} / 30")
        st.subheader("Frozen LLM risk ratings")
        st.dataframe(counts.reset_index(names="Control group"), use_container_width=True, hide_index=True)
        st.bar_chart(counts, y_label="Number of normal user-days")
        st.info(
            "30 flagged-normal controls received 28 Medium and 2 High ratings; "
            "all 30 unflagged-normal controls received Low. The frozen LLM can output Low, "
            "but it did not de-escalate the sampled detector-generated false positives. "
            "This does NOT demonstrate improved detection or reduced investigation workload."
        )
        st.caption(
            "Diagnostic only: unflagged cases are not sent to the LLM in the normal "
            "candidate-gated pipeline. True labels and diagnostic group names were "
            "not included in the model input. High is an investigation priority, not malicious-intent attribution."
        )
    except Exception as exc:
        st.error(f"Cannot load the LLM normal-control diagnostic: {exc}")
        st.caption("Expected file: outputs/robustness/llm_control_summary.csv")

with method_tab:
    st.header("Methodology, Scope and Interpretation")
    st.markdown(
        """
**Observation unit:** One synthetic employee-day, aggregated from raw audit events.

**Roles:** Teller, Customer Service and Manager; separate Isolation Forest model per role.

**Training / validation / testing:** Separate normal training data, unseen normal validation
for 95th-percentile role thresholds, independent injected test datasets.

**ML score:** `anomaly_score = -model.score_samples(X)`; higher means more unusual.
The built-in Isolation Forest cut-off was not used.

**Rules:** Late activity, large export and high record access; role-specific fixed
thresholds estimated using normal validation data.

**Candidate selection:** An employee-day enters the analyst queue if the Isolation
Forest OR at least one rule flags it. Ground-truth labels are used only after
inference for evaluation.

**Original controlled experiments:** 300 employee-days per experiment, 30 injected
anomalies, same selected cases at three strengths; individual candidate evidence
and 224 original LLM assessments can be inspected on the demo tab.

**Final robustness evaluation:** 50 independently generated test datasets per
condition (normal-only control / obvious / moderate / subtle), 200 datasets and
60,000 employee-days total. The trained models and thresholds remain fixed;
95% confidence intervals use Student's t distribution across 50 replications
(df=49). Different robustness conditions are not paired case-by-case.

**LLM:** The stored `gpt-5-nano-2025-08-07` / frozen prompt-v2 assessments provide
context, evidence summaries, uncertainty and suggested investigation priority.
The dashboard makes **no live LLM API calls**. The 60-case normal-control study
is a separate evaluation-only diagnostic, not a deployment workflow.

**Main limitations:** All roles, behaviour distributions and anomalies are synthetic;
rule thresholds reflect the synthetic validation distribution rather than banking
policy; injected scenarios directly affect rule-monitored features; a context-free
role baseline and a deterministic prioritisation baseline were not evaluated.
The LLM has not demonstrated reduction of detector false-positive workload.
An anomalous day is not proof of malicious intent, policy violation or compromise.
"""
    )
    st.caption("Source: submitted UTS 32040 final report, Sections 6–10 and Appendix B.")
    st.markdown("[View project source repository](https://github.com/fachrulreza20/Capstone-Cybersec)")

st.caption(
    "Final-report-aligned evidence dashboard — synthetic evaluation only; "
    "human-in-the-loop security investigation required."
)
