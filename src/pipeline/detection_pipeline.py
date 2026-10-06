import json
from pathlib import Path

import joblib
import pandas as pd

from src.features.build_user_day_features import build_user_day_features
from src.features.feature_config import ML_FEATURES
from src.rules.evaluate_rule_baseline import apply_rules


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"

ML_THRESHOLD_PATH = (
    MODEL_DIR / "role_thresholds.json"
)

RULE_THRESHOLD_PATH = (
    MODEL_DIR / "rule_thresholds.json"
)

LLM_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "llm_config.json"
)


KEY_COLUMNS = [
    "user_id",
    "date",
    "role",
]


FORBIDDEN_LLM_FIELDS = {
    "true_label",
    "case_id",
    "scenario",
    "difficulty",
    "experiment_difficulty",
}


ROLE_CONTEXT = {
    "Teller": (
        "Daily transaction-processing role with moderate customer-record "
        "access and normally limited data export activity."
    ),
    "Customer Service": (
        "Customer enquiry and account-maintenance role with relatively high "
        "customer-record access and limited data export activity."
    ),
    "Manager": (
        "Operational supervision role with moderate customer-record access "
        "and comparatively higher reporting and data export activity."
    ),
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_keys(df):
    df = df.copy()

    missing = [
        column
        for column in KEY_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing key columns: {missing}"
        )

    df["user_id"] = (
        df["user_id"]
        .astype(str)
        .str.strip()
    )

    df["role"] = (
        df["role"]
        .astype(str)
        .str.strip()
    )

    df["date"] = (
        pd.to_datetime(df["date"])
        .dt.strftime("%Y-%m-%d")
    )

    return df


def python_value(value):
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def validate_unique_keys(df, name):
    duplicates = df.duplicated(
        subset=KEY_COLUMNS
    ).sum()

    if duplicates:
        raise ValueError(
            f"{name}: found {duplicates} duplicate "
            f"user-day keys."
        )


def validate_no_forbidden_fields(data):
    if isinstance(data, dict):
        for key, value in data.items():

            if key in FORBIDDEN_LLM_FIELDS:
                raise ValueError(
                    "Forbidden field leaked into "
                    f"LLM context: {key}"
                )

            validate_no_forbidden_fields(value)

    elif isinstance(data, list):
        for item in data:
            validate_no_forbidden_fields(item)


# ============================================================
# LOAD FROZEN CONFIGURATION
# ============================================================

def load_ml_thresholds():
    with open(
        ML_THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if "roles" not in data:
        raise ValueError(
            "Invalid ML threshold file: "
            "'roles' section missing."
        )

    thresholds = {}

    for role, role_info in data["roles"].items():

        if "threshold" not in role_info:
            raise ValueError(
                f"ML threshold missing for role: {role}"
            )

        thresholds[role] = float(
            role_info["threshold"]
        )

    return thresholds


def load_rule_thresholds():
    with open(
        RULE_THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_llm_config():
    with open(
        LLM_CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    if config.get("include_raw_timeline") is not False:
        raise ValueError(
            "LLM config must keep "
            "include_raw_timeline=false."
        )

    return config


# ============================================================
# MODEL LOADING
# ============================================================

def model_filename(role):
    safe_role = (
        role
        .lower()
        .replace(" ", "_")
    )

    return (
        MODEL_DIR
        / f"isolation_forest_{safe_role}.joblib"
    )


def load_role_model(role):
    path = model_filename(role)

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found for {role}: {path}"
        )

    package = joblib.load(path)

    if "model" not in package:
        raise ValueError(
            f"Model package for {role} "
            "does not contain 'model'."
        )

    if "features" not in package:
        raise ValueError(
            f"Model package for {role} "
            "does not contain 'features'."
        )

    if list(package["features"]) != list(ML_FEATURES):
        raise ValueError(
            f"Feature mismatch for {role}.\n"
            f"Model: {package['features']}\n"
            f"Runtime: {ML_FEATURES}"
        )

    return package["model"]


# ============================================================
# STAGE 1 — FEATURE ENGINEERING
# ============================================================

def build_features_from_raw(raw_df):
    required_raw_columns = {
        "timestamp",
        "user_id",
        "role",
        "event_type",
        "account_type",
        "failed_login",
        "download_count",
        "records_accessed",
        "is_known_ip",
        "ip_address",
    }

    missing = sorted(
        required_raw_columns
        - set(raw_df.columns)
    )

    if missing:
        raise ValueError(
            f"Raw audit log missing columns: {missing}"
        )

    features = build_user_day_features(
        raw_df
    )

    features = normalize_keys(features)

    validate_unique_keys(
        features,
        "Feature dataset",
    )

    if features.empty:
        raise ValueError(
            "Feature engineering produced zero user-days."
        )

    return features


# ============================================================
# STAGE 2 — ROLE-SPECIFIC ISOLATION FOREST
# ============================================================

def score_ml(features):
    df = normalize_keys(features)

    validate_unique_keys(
        df,
        "ML input",
    )

    missing_features = [
        feature
        for feature in ML_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing ML features: {missing_features}"
        )

    if df[ML_FEATURES].isna().any().any():
        raise ValueError(
            "Missing values found in ML features."
        )

    thresholds = load_ml_thresholds()

    result_parts = []

    for role, role_df in df.groupby(
        "role",
        sort=True,
    ):

        if role not in thresholds:
            raise ValueError(
                "No frozen ML threshold found "
                f"for role: {role}"
            )

        role_df = role_df.copy()

        model = load_role_model(role)

        X = role_df[ML_FEATURES]

        # Frozen project definition:
        #
        # sklearn score_samples:
        # lower = more abnormal
        #
        # project anomaly_score:
        # -score_samples
        #
        # therefore higher = more unusual

        raw_score = model.score_samples(X)

        role_df["anomaly_score"] = (
            -raw_score
        )

        role_df["threshold"] = (
            thresholds[role]
        )

        role_df["prediction"] = (
            role_df["anomaly_score"]
            > role_df["threshold"]
        ).astype(int)

        result_parts.append(role_df)

    result = pd.concat(
        result_parts,
        ignore_index=True,
    )

    result = result.sort_values(
        KEY_COLUMNS
    ).reset_index(drop=True)

    return result


# ============================================================
# STAGE 3 — DETERMINISTIC RULES
# ============================================================

def score_rules(features):
    df = normalize_keys(features)

    validate_unique_keys(
        df,
        "Rule input",
    )

    threshold_data = load_rule_thresholds()

    result = apply_rules(
        df,
        threshold_data,
    )

    result = normalize_keys(result)

    result = result.sort_values(
        KEY_COLUMNS
    ).reset_index(drop=True)

    return result


# ============================================================
# STAGE 4 — COMBINE DETECTOR EVIDENCE
# ============================================================

def combine_detector_results(
    features,
    ml_results,
    rule_results,
):
    features = normalize_keys(features)
    ml_results = normalize_keys(ml_results)
    rule_results = normalize_keys(rule_results)

    ml_columns = KEY_COLUMNS + [
        "anomaly_score",
        "threshold",
        "prediction",
    ]

    rule_columns = KEY_COLUMNS + [
        "rule_after_hours",
        "rule_large_export",
        "rule_high_record_access",
        "rules_triggered",
        "rule_prediction",
    ]

    missing_ml = [
        column
        for column in ml_columns
        if column not in ml_results.columns
    ]

    missing_rules = [
        column
        for column in rule_columns
        if column not in rule_results.columns
    ]

    if missing_ml:
        raise ValueError(
            f"Missing ML result columns: {missing_ml}"
        )

    if missing_rules:
        raise ValueError(
            f"Missing rule result columns: {missing_rules}"
        )

    combined = features.merge(
        ml_results[ml_columns],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    combined = combined.merge(
        rule_results[rule_columns],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    if len(combined) != len(features):
        raise ValueError(
            "Detector merge changed row count."
        )

    combined["candidate"] = (
        (combined["prediction"].astype(int) == 1)
        |
        (
            combined["rule_prediction"]
            .astype(int)
            == 1
        )
    ).astype(int)

    return combined


# ============================================================
# STAGE 5 — BUILD LLM-READY CANDIDATE CONTEXT
# ============================================================

def build_candidate_packages(combined):
    config = load_llm_config()

    feature_names = config[
        "include_behaviour_features"
    ]

    candidates = combined[
        combined["candidate"] == 1
    ].copy()

    packages = []

    for _, row in candidates.iterrows():

        behaviour = {}

        for feature in feature_names:

            if feature not in row.index:
                raise ValueError(
                    "Configured LLM behaviour feature "
                    f"not found: {feature}"
                )

            behaviour[feature] = python_value(
                row[feature]
            )

        package = {
            "employee_context": {
                "user_id": row["user_id"],
                "date": row["date"],
                "role": row["role"],
                "role_context": ROLE_CONTEXT.get(
                    row["role"],
                    "No role context available.",
                ),
            },

            "behaviour_summary": behaviour,

            "ml_evidence": {
                "anomaly_score": python_value(
                    row["anomaly_score"]
                ),
                "threshold": python_value(
                    row["threshold"]
                ),
                "prediction": (
                    "Anomaly"
                    if int(row["prediction"]) == 1
                    else "Normal"
                ),
            },

            "rule_evidence": {
                "after_hours": bool(
                    int(row["rule_after_hours"])
                ),
                "large_export": bool(
                    int(row["rule_large_export"])
                ),
                "high_record_access": bool(
                    int(
                        row[
                            "rule_high_record_access"
                        ]
                    )
                ),
                "rules_triggered": int(
                    row["rules_triggered"]
                ),
                "prediction": (
                    "Anomaly"
                    if int(
                        row["rule_prediction"]
                    ) == 1
                    else "Normal"
                ),
            },
        }

        validate_no_forbidden_fields(
            package
        )

        packages.append(package)

    return packages


# ============================================================
# STAGE 6 — ANALYST-FACING DETECTOR TABLE
# ============================================================

def build_analyst_table(combined):
    columns = [
        "user_id",
        "date",
        "role",
        "anomaly_score",
        "threshold",
        "prediction",
        "rule_after_hours",
        "rule_large_export",
        "rule_high_record_access",
        "rules_triggered",
        "rule_prediction",
        "candidate",
    ]

    table = combined[
        columns
    ].copy()

    table = table.rename(
        columns={
            "prediction": "ml_prediction",
        }
    )

    table["ml_status"] = table[
        "ml_prediction"
    ].map(
        {
            0: "Normal",
            1: "Anomaly",
        }
    )

    table["rule_status"] = table[
        "rule_prediction"
    ].map(
        {
            0: "Normal",
            1: "Anomaly",
        }
    )

    table["investigation_candidate"] = (
        table["candidate"]
        .map(
            {
                0: "No",
                1: "Yes",
            }
        )
    )

    return table


# ============================================================
# COMPLETE PIPELINE — FEATURES INPUT
# ============================================================

def run_feature_pipeline(features):
    features = normalize_keys(features)

    ml_results = score_ml(features)

    rule_results = score_rules(features)

    combined = combine_detector_results(
        features=features,
        ml_results=ml_results,
        rule_results=rule_results,
    )

    candidate_packages = (
        build_candidate_packages(
            combined
        )
    )

    analyst_table = build_analyst_table(
        combined
    )

    return {
        "features": features,
        "ml_results": ml_results,
        "rule_results": rule_results,
        "combined": combined,
        "candidate_packages": candidate_packages,
        "analyst_table": analyst_table,
    }


# ============================================================
# COMPLETE PIPELINE — RAW AUDIT LOG INPUT
# ============================================================

def run_raw_pipeline(raw_df):
    features = build_features_from_raw(
        raw_df
    )

    result = run_feature_pipeline(
        features
    )

    result["raw_event_count"] = len(
        raw_df
    )

    return result