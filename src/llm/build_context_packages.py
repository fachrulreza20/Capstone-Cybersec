import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "configs" / "llm_config.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "llm_context"

EXPERIMENTS = [
    "experiment_1",
    "experiment_2",
    "experiment_3",
]

KEY_COLUMNS = ["user_id", "date", "role"]

FORBIDDEN_FIELDS = {
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


def normalize_keys(df):
    df = df.copy()

    required = ["user_id", "date", "role"]

    missing = [
        col for col in required
        if col not in df.columns
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


def validate_no_forbidden_fields(data):

    if isinstance(data, dict):

        for key, value in data.items():

            if key in FORBIDDEN_FIELDS:
                raise ValueError(
                    f"Forbidden field leaked into LLM context: {key}"
                )

            validate_no_forbidden_fields(value)

    elif isinstance(data, list):

        for item in data:
            validate_no_forbidden_fields(item)


def build_experiment_context(experiment, config):

    print(f"\n=== Building {experiment} ===")

    feature_path = (
        PROCESSED_DIR
        / f"{experiment}_user_day.csv"
    )

    ml_path = (
        PREDICTION_DIR
        / f"{experiment}_predictions.csv"
    )

    rule_path = (
        PREDICTION_DIR
        / f"{experiment}_rule_evaluated.csv"
    )

    for path in [
        feature_path,
        ml_path,
        rule_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    features = normalize_keys(
        pd.read_csv(feature_path)
    )

    ml = normalize_keys(
        pd.read_csv(ml_path)
    )

    rules = normalize_keys(
        pd.read_csv(rule_path)
    )

    # --------------------------------------------------
    # Integrity checks
    # --------------------------------------------------

    for name, df in [
        ("features", features),
        ("ML predictions", ml),
        ("rule results", rules),
    ]:

        if len(df) != 300:
            raise ValueError(
                f"{experiment}: expected 300 rows in "
                f"{name}, found {len(df)}"
            )

        duplicates = df.duplicated(
            subset=KEY_COLUMNS
        ).sum()

        if duplicates:
            raise ValueError(
                f"{experiment}: duplicate keys in {name}"
            )

    # --------------------------------------------------
    # Required detector columns
    # --------------------------------------------------

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
        c for c in ml_columns
        if c not in ml.columns
    ]

    missing_rules = [
        c for c in rule_columns
        if c not in rules.columns
    ]

    if missing_ml:
        raise ValueError(
            f"Missing ML columns: {missing_ml}"
        )

    if missing_rules:
        raise ValueError(
            f"Missing rule columns: {missing_rules}"
        )

    # --------------------------------------------------
    # Merge WITHOUT ground truth
    # --------------------------------------------------

    combined = features.merge(
        ml[ml_columns],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    combined = combined.merge(
        rules[rule_columns],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    if len(combined) != 300:
        raise ValueError(
            f"{experiment}: expected 300 rows after merge, "
            f"found {len(combined)}"
        )

    # --------------------------------------------------
    # Candidate = ML anomaly OR rule anomaly
    # --------------------------------------------------

    candidates = combined[
        (combined["prediction"].astype(int) == 1)
        |
        (combined["rule_prediction"].astype(int) == 1)
    ].copy()

    print(f"Total user-days : {len(combined)}")
    print(f"LLM candidates  : {len(candidates)}")

    feature_names = config[
        "include_behaviour_features"
    ]

    packages = []

    # --------------------------------------------------
    # Build structured LLM input
    # --------------------------------------------------

    for _, row in candidates.iterrows():

        behaviour = {}

        for feature in feature_names:

            if feature not in row.index:
                raise ValueError(
                    f"Configured feature not found: {feature}"
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
                    int(row["rule_high_record_access"])
                ),
                "rules_triggered": int(
                    row["rules_triggered"]
                ),
                "prediction": (
                    "Anomaly"
                    if int(row["rule_prediction"]) == 1
                    else "Normal"
                ),
            },
        }

        validate_no_forbidden_fields(package)

        packages.append(package)

    output = {
        "experiment": experiment,

        "candidate_selection": (
            "Isolation Forest anomaly OR "
            "deterministic rule anomaly"
        ),

        "input_design": (
            "Structured user-day behavioural context. "
            "Raw event timeline excluded."
        ),

        "candidate_count": len(packages),

        "candidates": packages,
    }

    validate_no_forbidden_fields(output)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{experiment}_context.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved: {output_path}")

    return len(packages)


def main():

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config not found: {CONFIG_PATH}"
        )

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    if config.get("include_raw_timeline") is not False:
        raise ValueError(
            "LLM config must set include_raw_timeline to false."
        )

    print("=== Build Final LLM Context Packages ===")

    counts = {}

    for experiment in EXPERIMENTS:

        counts[experiment] = (
            build_experiment_context(
                experiment,
                config,
            )
        )

    print("\n=== Summary ===")

    for experiment, count in counts.items():
        print(
            f"{experiment}: {count} candidates"
        )

    print("\nRaw timeline included : NO")
    print("Ground-truth leakage  : PASS")
    print("Context package build : PASS")


if __name__ == "__main__":
    main()