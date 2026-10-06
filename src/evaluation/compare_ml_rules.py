from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
)

EXPERIMENTS = {
    "experiment_1": "obvious",
    "experiment_2": "moderate",
    "experiment_3": "subtle",
}

KEY_COLUMNS = [
    "user_id",
    "date",
    "role",
]


def normalize_keys(df):
    df = df.copy()

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


def classify_overlap(row):
    ml = int(row["prediction"])
    rule = int(row["rule_prediction"])

    if ml == 1 and rule == 1:
        return "BOTH"

    if ml == 0 and rule == 1:
        return "RULE_ONLY"

    if ml == 1 and rule == 0:
        return "ML_ONLY"

    return "MISSED_BY_BOTH"


def analyse_experiment(
    experiment_name,
    difficulty,
):
    ml_path = (
        PREDICTION_DIR
        / f"{experiment_name}_evaluated.csv"
    )

    rule_path = (
        PREDICTION_DIR
        / f"{experiment_name}_rule_evaluated.csv"
    )

    if not ml_path.exists():
        raise FileNotFoundError(
            f"ML evaluated file not found: {ml_path}"
        )

    if not rule_path.exists():
        raise FileNotFoundError(
            f"Rule evaluated file not found: {rule_path}"
        )

    ml = normalize_keys(
        pd.read_csv(ml_path)
    )

    rules = normalize_keys(
        pd.read_csv(rule_path)
    )

    if len(ml) != 300:
        raise ValueError(
            f"{experiment_name}: "
            f"ML file should contain 300 rows."
        )

    if len(rules) != 300:
        raise ValueError(
            f"{experiment_name}: "
            f"rule file should contain 300 rows."
        )

    # --------------------------------------------------------
    # Keep only rule-specific columns from rule result.
    # Ground truth and scenario metadata come from ML file.
    # --------------------------------------------------------

    rule_columns = (
        KEY_COLUMNS
        + [
            "rule_after_hours",
            "rule_large_export",
            "rule_high_record_access",
            "rules_triggered",
            "rule_prediction",
        ]
    )

    missing_rule_columns = [
        column
        for column in rule_columns
        if column not in rules.columns
    ]

    if missing_rule_columns:
        raise ValueError(
            f"{experiment_name}: missing rule columns "
            f"{missing_rule_columns}"
        )

    combined = ml.merge(
        rules[rule_columns],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    if len(combined) != 300:
        raise ValueError(
            f"{experiment_name}: "
            "ML/rule merge did not produce 300 rows."
        )

    # --------------------------------------------------------
    # Verify both files refer to the same experiment population.
    # --------------------------------------------------------

    if int(combined["true_label"].sum()) != 30:
        raise ValueError(
            f"{experiment_name}: "
            "expected exactly 30 true anomalies."
        )

    combined["experiment"] = (
        experiment_name
    )

    combined["experiment_difficulty"] = (
        difficulty
    )

    # --------------------------------------------------------
    # Analyse injected anomalies
    # --------------------------------------------------------

    anomalies = combined[
        combined["true_label"] == 1
    ].copy()

    anomalies["detection_overlap"] = (
        anomalies.apply(
            classify_overlap,
            axis=1,
        )
    )

    overlap_order = [
        "BOTH",
        "RULE_ONLY",
        "ML_ONLY",
        "MISSED_BY_BOTH",
    ]

    overlap_counts = (
        anomalies[
            "detection_overlap"
        ]
        .value_counts()
        .reindex(
            overlap_order,
            fill_value=0,
        )
    )

    # --------------------------------------------------------
    # Scenario overlap
    # --------------------------------------------------------

    scenario_rows = []

    if "scenario" not in anomalies.columns:
        raise ValueError(
            f"{experiment_name}: "
            "scenario column not found."
        )

    for scenario, group in anomalies.groupby(
        "scenario"
    ):
        counts = (
            group["detection_overlap"]
            .value_counts()
        )

        scenario_rows.append({
            "experiment": experiment_name,
            "difficulty": difficulty,
            "scenario": scenario,
            "anomaly_cases": len(group),
            "both": int(
                counts.get("BOTH", 0)
            ),
            "rule_only": int(
                counts.get("RULE_ONLY", 0)
            ),
            "ml_only": int(
                counts.get("ML_ONLY", 0)
            ),
            "missed_by_both": int(
                counts.get(
                    "MISSED_BY_BOTH",
                    0,
                )
            ),
        })

    # --------------------------------------------------------
    # Normal-case false-positive overlap
    # --------------------------------------------------------

    normal = combined[
        combined["true_label"] == 0
    ].copy()

    normal["ml_fp"] = (
        normal["prediction"] == 1
    )

    normal["rule_fp"] = (
        normal["rule_prediction"] == 1
    )

    fp_both = int(
        (
            normal["ml_fp"]
            & normal["rule_fp"]
        ).sum()
    )

    fp_ml_only = int(
        (
            normal["ml_fp"]
            & ~normal["rule_fp"]
        ).sum()
    )

    fp_rule_only = int(
        (
            ~normal["ml_fp"]
            & normal["rule_fp"]
        ).sum()
    )

    fp_neither = int(
        (
            ~normal["ml_fp"]
            & ~normal["rule_fp"]
        ).sum()
    )

    summary = {
        "experiment": experiment_name,
        "difficulty": difficulty,
        "true_anomalies": len(anomalies),
        "both_detected": int(
            overlap_counts["BOTH"]
        ),
        "rule_only": int(
            overlap_counts["RULE_ONLY"]
        ),
        "ml_only": int(
            overlap_counts["ML_ONLY"]
        ),
        "missed_by_both": int(
            overlap_counts["MISSED_BY_BOTH"]
        ),
        "normal_cases": len(normal),
        "fp_both": fp_both,
        "fp_ml_only": fp_ml_only,
        "fp_rule_only": fp_rule_only,
        "normal_neither_flagged": fp_neither,
    }

    # --------------------------------------------------------
    # Save detailed case comparison
    # --------------------------------------------------------

    preferred_columns = [
        "experiment",
        "experiment_difficulty",
        "case_id",
        "user_id",
        "date",
        "role",
        "scenario",
        "anomaly_score",
        "threshold",
        "prediction",
        "rule_after_hours",
        "rule_large_export",
        "rule_high_record_access",
        "rules_triggered",
        "rule_prediction",
        "detection_overlap",
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in anomalies.columns
    ]

    anomaly_output = anomalies[
        available_columns
    ].copy()

    return (
        summary,
        scenario_rows,
        anomaly_output,
    )


def main():

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=== ML vs Rule Overlap Analysis ==="
    )

    summary_rows = []
    scenario_rows = []
    case_rows = []

    for experiment, difficulty in (
        EXPERIMENTS.items()
    ):

        (
            summary,
            experiment_scenarios,
            anomaly_cases,
        ) = analyse_experiment(
            experiment,
            difficulty,
        )

        summary_rows.append(summary)

        scenario_rows.extend(
            experiment_scenarios
        )

        case_rows.append(
            anomaly_cases
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    scenario_df = pd.DataFrame(
        scenario_rows
    )

    cases_df = pd.concat(
        case_rows,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Save evidence
    # --------------------------------------------------------

    summary_path = (
        METRICS_DIR
        / "ml_rule_overlap_summary.csv"
    )

    scenario_path = (
        METRICS_DIR
        / "ml_rule_overlap_by_scenario.csv"
    )

    cases_path = (
        METRICS_DIR
        / "ml_rule_overlap_cases.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    scenario_df.to_csv(
        scenario_path,
        index=False,
    )

    cases_df.to_csv(
        cases_path,
        index=False,
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print(
        "\n=== Anomaly Detection Overlap ==="
    )

    print(
        summary_df[
            [
                "experiment",
                "difficulty",
                "both_detected",
                "rule_only",
                "ml_only",
                "missed_by_both",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== False Positive Overlap ==="
    )

    print(
        summary_df[
            [
                "experiment",
                "difficulty",
                "fp_both",
                "fp_ml_only",
                "fp_rule_only",
                "normal_neither_flagged",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== Overlap by Scenario ==="
    )

    print(
        scenario_df.to_string(
            index=False
        )
    )

    print(
        "\n=== Files Saved ==="
    )

    print(summary_path)
    print(scenario_path)
    print(cases_path)


if __name__ == "__main__":
    main()