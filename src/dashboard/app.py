import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.pipeline.detection_pipeline import run_raw_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
LLM_RESULTS_DIR = PROJECT_ROOT / "outputs" / "llm_results"


EXPERIMENTS = {
    "Experiment 1 — Obvious": {
        "raw_file": RAW_DIR / "experiment_1_test.csv",
        "llm_file": LLM_RESULTS_DIR / "experiment_1_llm_results.json",
        "difficulty": "Obvious",
    },
    "Experiment 2 — Moderate": {
        "raw_file": RAW_DIR / "experiment_2_test.csv",
        "llm_file": LLM_RESULTS_DIR / "experiment_2_llm_results.json",
        "difficulty": "Moderate",
    },
    "Experiment 3 — Subtle": {
        "raw_file": RAW_DIR / "experiment_3_test.csv",
        "llm_file": LLM_RESULTS_DIR / "experiment_3_llm_results.json",
        "difficulty": "Subtle",
    },
}


st.set_page_config(
    page_title="Banking Behavioural Anomaly Detection",
    page_icon="🏦",
    layout="wide",
)


# ============================================================
# HELPERS
# ============================================================

def normalize_date(value):
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def candidate_key(user_id, date, role):
    return (
        str(user_id).strip(),
        normalize_date(date),
        str(role).strip(),
    )


@st.cache_data
def load_raw_csv(path_string):
    path = Path(path_string)

    if not path.exists():
        raise FileNotFoundError(
            f"Raw audit-log file not found: {path}"
        )

    return pd.read_csv(
        path,
        parse_dates=["timestamp"],
    )


@st.cache_data
def load_llm_results(path_string):
    path = Path(path_string)

    if not path.exists():
        return {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return data


def extract_llm_index(data):
    """
    Build a lookup:
    (user_id, date, role) -> assessment

    The frozen LLM result files may store completed assessments
    in slightly different wrapper structures, so this function
    searches recursively for records containing both candidate
    identity/context and assessment output.
    """

    index = {}

    def walk(obj):
        if isinstance(obj, dict):

            assessment = None

            if isinstance(
                obj.get("assessment"),
                dict,
            ):
                assessment = obj["assessment"]

            elif (
                "risk_level" in obj
                and
                "investigation_priority" in obj
            ):
                assessment = obj

            candidate = None

            if isinstance(
                obj.get("candidate"),
                dict,
            ):
                candidate = obj["candidate"]

            if candidate is None:
                candidate = obj

            employee_context = None

            if isinstance(candidate, dict):
                if isinstance(
                    candidate.get(
                        "employee_context"
                    ),
                    dict,
                ):
                    employee_context = candidate[
                        "employee_context"
                    ]

            if (
                assessment is not None
                and employee_context is not None
            ):
                user_id = employee_context.get(
                    "user_id"
                )
                date = employee_context.get(
                    "date"
                )
                role = employee_context.get(
                    "role"
                )

                if (
                    user_id is not None
                    and date is not None
                    and role is not None
                ):
                    key = candidate_key(
                        user_id,
                        date,
                        role,
                    )

                    index[key] = assessment

            for value in obj.values():
                walk(value)

        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)

    return index


def format_status(value):
    return (
        "Anomaly"
        if int(value) == 1
        else "Normal"
    )


def yes_no(value):
    return "Yes" if bool(value) else "No"


def render_llm_assessment(assessment):
    if not assessment:
        st.info(
            "No frozen LLM assessment was found "
            "for this candidate."
        )
        return

    risk = assessment.get(
        "risk_level",
        "Not available",
    )

    priority = assessment.get(
        "investigation_priority",
        "Not available",
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Contextual Risk",
        risk,
    )

    col2.metric(
        "Investigation Priority",
        priority,
    )

    st.markdown("#### Summary")

    st.write(
        assessment.get(
            "summary",
            "Not available.",
        )
    )

    st.markdown("#### Evidence")

    evidence = assessment.get(
        "evidence",
        [],
    )

    if evidence:
        for item in evidence:
            st.markdown(f"- {item}")
    else:
        st.write("No evidence text available.")

    st.markdown("#### Role Context")

    st.write(
        assessment.get(
            "role_context",
            "Not available.",
        )
    )

    st.markdown("#### Uncertainty")

    st.write(
        assessment.get(
            "uncertainty",
            "Not available.",
        )
    )

    st.markdown("#### Recommended Action")

    st.write(
        assessment.get(
            "recommended_action",
            "Not available.",
        )
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "Banking Behavioural Anomaly Detection"
)

st.caption(
    "Role-specific behavioural anomaly detection "
    "with deterministic rules and contextual "
    "LLM-assisted investigation support."
)

st.warning(
    "Research prototype using synthetic banking audit logs. "
    "An anomaly indicates unusual behaviour, not malicious "
    "intent or confirmed compromise. Human investigation "
    "is required."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Detection Input")

selected_experiment = st.sidebar.selectbox(
    "Synthetic experiment",
    list(EXPERIMENTS.keys()),
)

experiment_config = EXPERIMENTS[
    selected_experiment
]

st.sidebar.markdown(
    f"**Difficulty:** "
    f"{experiment_config['difficulty']}"
)

st.sidebar.markdown(
    "**Candidate logic:** ML anomaly OR rule anomaly"
)

st.sidebar.markdown(
    "**LLM mode:** Frozen experimental assessment"
)

st.sidebar.info(
    "The dashboard does not call the OpenAI API. "
    "Existing frozen LLM assessments are displayed "
    "for reproducibility."
)


# ============================================================
# LOAD + RUN PIPELINE
# ============================================================

try:
    raw_df = load_raw_csv(
        str(experiment_config["raw_file"])
    )

    result = run_raw_pipeline(
        raw_df.copy()
    )

except Exception as exc:
    st.error(
        f"Pipeline failed: {exc}"
    )
    st.stop()


features = result["features"]
combined = result["combined"].copy()
candidate_packages = result[
    "candidate_packages"
]


# ============================================================
# LOAD FROZEN LLM RESULTS
# ============================================================

llm_data = load_llm_results(
    str(experiment_config["llm_file"])
)

llm_index = extract_llm_index(
    llm_data
)


# ============================================================
# SUMMARY METRICS
# ============================================================

ml_anomalies = int(
    combined["prediction"].sum()
)

rule_anomalies = int(
    combined["rule_prediction"].sum()
)

candidate_count = int(
    combined["candidate"].sum()
)

st.subheader("Detection Summary")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Raw Events",
    f"{len(raw_df):,}",
)

col2.metric(
    "User-Days",
    f"{len(features):,}",
)

col3.metric(
    "ML Anomalies",
    ml_anomalies,
)

col4.metric(
    "Rule Anomalies",
    rule_anomalies,
)

col5.metric(
    "Investigation Candidates",
    candidate_count,
)


# ============================================================
# ROLE SUMMARY
# ============================================================

st.subheader("Role-Level Detection Summary")

role_summary = (
    combined
    .groupby("role")
    .agg(
        user_days=("user_id", "size"),
        ml_anomalies=("prediction", "sum"),
        rule_anomalies=(
            "rule_prediction",
            "sum",
        ),
        candidates=("candidate", "sum"),
    )
    .reset_index()
)

st.dataframe(
    role_summary,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# CANDIDATE TABLE
# ============================================================

st.subheader("Investigation Candidates")

candidates = combined[
    combined["candidate"] == 1
].copy()

candidates["ML Status"] = candidates[
    "prediction"
].apply(format_status)

candidates["Rule Status"] = candidates[
    "rule_prediction"
].apply(format_status)

candidate_display = candidates[
    [
        "user_id",
        "date",
        "role",
        "ML Status",
        "anomaly_score",
        "threshold",
        "Rule Status",
        "rules_triggered",
    ]
].copy()

candidate_display = candidate_display.rename(
    columns={
        "user_id": "User",
        "date": "Date",
        "role": "Role",
        "anomaly_score": "ML Anomaly Score",
        "threshold": "ML Threshold",
        "rules_triggered": "Rules Triggered",
    }
)

st.dataframe(
    candidate_display,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# CANDIDATE SELECTION
# ============================================================

st.subheader("Candidate Investigation")

if candidates.empty:
    st.success(
        "No investigation candidates were identified."
    )
    st.stop()


candidate_options = []

candidate_lookup = {}

for _, row in candidates.iterrows():

    key = candidate_key(
        row["user_id"],
        row["date"],
        row["role"],
    )

    label = (
        f"{row['user_id']} | "
        f"{normalize_date(row['date'])} | "
        f"{row['role']}"
    )

    candidate_options.append(label)
    candidate_lookup[label] = key


selected_candidate_label = st.selectbox(
    "Select employee-day",
    candidate_options,
)

selected_key = candidate_lookup[
    selected_candidate_label
]

selected_user_id, selected_date, selected_role = (
    selected_key
)

selected_rows = candidates[
    (
        candidates["user_id"]
        .astype(str)
        .str.strip()
        == selected_user_id
    )
    &
    (
        pd.to_datetime(
            candidates["date"]
        )
        .dt.strftime("%Y-%m-%d")
        == selected_date
    )
    &
    (
        candidates["role"]
        .astype(str)
        .str.strip()
        == selected_role
    )
]

if len(selected_rows) != 1:
    st.error(
        "Selected candidate could not be resolved "
        "to exactly one user-day."
    )
    st.stop()

row = selected_rows.iloc[0]


# ============================================================
# CANDIDATE OVERVIEW
# ============================================================

st.markdown("### Employee-Day Context")

context_col1, context_col2, context_col3 = (
    st.columns(3)
)

context_col1.metric(
    "Employee",
    row["user_id"],
)

context_col2.metric(
    "Role",
    row["role"],
)

context_col3.metric(
    "Date",
    normalize_date(row["date"]),
)


# ============================================================
# ML EVIDENCE
# ============================================================

st.markdown("### ML Evidence")

ml_col1, ml_col2, ml_col3 = st.columns(3)

ml_col1.metric(
    "ML Prediction",
    format_status(
        row["prediction"]
    ),
)

ml_col2.metric(
    "Anomaly Score",
    f"{row['anomaly_score']:.6f}",
)

ml_col3.metric(
    "Role Threshold",
    f"{row['threshold']:.6f}",
)

if int(row["prediction"]) == 1:
    st.write(
        "The role-specific Isolation Forest anomaly "
        "score exceeded the frozen role threshold."
    )
else:
    st.write(
        "The role-specific Isolation Forest anomaly "
        "score did not exceed the frozen role threshold."
    )


# ============================================================
# RULE EVIDENCE
# ============================================================

st.markdown("### Deterministic Rule Evidence")

rule_col1, rule_col2, rule_col3, rule_col4 = (
    st.columns(4)
)

rule_col1.metric(
    "After Hours",
    yes_no(
        int(row["rule_after_hours"])
    ),
)

rule_col2.metric(
    "Large Export",
    yes_no(
        int(row["rule_large_export"])
    ),
)

rule_col3.metric(
    "High Record Access",
    yes_no(
        int(
            row[
                "rule_high_record_access"
            ]
        )
    ),
)

rule_col4.metric(
    "Rules Triggered",
    int(row["rules_triggered"]),
)


# ============================================================
# BEHAVIOURAL FEATURES
# ============================================================

st.markdown("### Behavioural Summary")

behaviour_columns = [
    "event_count",
    "transaction_count",
    "customer_record_access_count",
    "download_event_count",
    "download_total",
    "records_accessed_total",
    "vip_access_count",
    "failed_login_count",
    "unknown_ip_count",
    "unique_ip_count",
    "first_activity_hour",
    "last_activity_hour",
    "activity_duration_hours",
]

available_behaviour_columns = [
    column
    for column in behaviour_columns
    if column in row.index
]

behaviour_table = pd.DataFrame(
    {
        "Feature": available_behaviour_columns,
        "Value": [
            row[column]
            for column
            in available_behaviour_columns
        ],
    }
)

st.dataframe(
    behaviour_table,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# LLM ASSESSMENT
# ============================================================

st.markdown(
    "### Contextual LLM Assessment"
)

assessment = llm_index.get(
    selected_key
)

render_llm_assessment(
    assessment
)


# ============================================================
# METHODOLOGY
# ============================================================

with st.expander(
    "How this prototype works"
):
    st.markdown(
        """
1. Synthetic raw banking audit events are aggregated into
   one behavioural observation per employee per day.

2. A role-specific Isolation Forest compares the employee-day
   against behavioural patterns learned separately for Teller,
   Customer Service, and Manager roles.

3. The ML anomaly score is compared with a frozen role-specific
   threshold calibrated using independent normal validation data.

4. Deterministic role-specific rules independently evaluate
   after-hours activity, large export volume, and high customer
   record access.

5. A user-day becomes an investigation candidate when either
   the ML detector or deterministic rules identify unusual
   behaviour.

6. The contextual LLM assessment is used to explain and
   prioritise surfaced candidates. It does not determine
   malicious intent and cannot recover anomalies missed by
   both upstream detectors.
        """
    )


st.caption(
    "Synthetic research prototype — intended for "
    "human-in-the-loop security investigation."
)