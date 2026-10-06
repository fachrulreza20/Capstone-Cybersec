import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

THRESHOLD_PATH = (
    PROJECT_ROOT
    / "models"
    / "rule_thresholds.json"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

GROUND_TRUTH_DIR = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
)

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


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0

    return numerator / denominator


def load_thresholds():
    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def apply_rules(df, threshold_data):
    results = []

    for role, role_df in df.groupby("role"):

        if role not in threshold_data["roles"]:
            raise ValueError(
                f"No rule thresholds found for role: {role}"
            )

        role_df = role_df.copy()

        thresholds = threshold_data[
            "roles"
        ][role]

        after_hours_threshold = (
            thresholds["after_hours"]["threshold"]
        )

        export_threshold = (
            thresholds["large_export"]["threshold"]
        )

        record_threshold = (
            thresholds[
                "high_record_access"
            ]["threshold"]
        )

        role_df["rule_after_hours"] = (
            role_df["last_activity_hour"]
            > after_hours_threshold
        ).astype(int)

        role_df["rule_large_export"] = (
            role_df["download_total"]
            > export_threshold
        ).astype(int)

        role_df[
            "rule_high_record_access"
        ] = (
            role_df["records_accessed_total"]
            > record_threshold
        ).astype(int)

        rule_columns = [
            "rule_after_hours",
            "rule_large_export",
            "rule_high_record_access",
        ]

        role_df["rules_triggered"] = (
            role_df[rule_columns]
            .sum(axis=1)
        )

        role_df["rule_prediction"] = (
            role_df["rules_triggered"] >= 1
        ).astype(int)

        results.append(role_df)

    return pd.concat(
        results,
        ignore_index=True,
    )


def evaluate(experiment_name, difficulty, thresholds):

    feature_path = (
        PROCESSED_DIR
        / f"{experiment_name}_user_day.csv"
    )

    ground_truth_path = (
        GROUND_TRUTH_DIR
        / f"{experiment_name}_ground_truth.csv"
    )

    features = pd.read_csv(
        feature_path
    )

    ground_truth = pd.read_csv(
        ground_truth_path
    )

    features = normalize_keys(features)
    ground_truth = normalize_keys(
        ground_truth
    )

    if len(features) != 300:
        raise ValueError(
            f"{experiment_name}: "
            f"expected 300 user-days, "
            f"found {len(features)}"
        )

    if len(ground_truth) != 30:
        raise ValueError(
            f"{experiment_name}: "
            f"expected 30 anomaly cases, "
            f"found {len(ground_truth)}"
        )

    scored = apply_rules(
        features,
        thresholds,
    )

    # --------------------------------------------------------
    # Build full ground truth
    # --------------------------------------------------------

    gt_columns = KEY_COLUMNS.copy()

    for optional_column in [
        "case_id",
        "scenario",
        "difficulty",
    ]:
        if optional_column in ground_truth.columns:
            gt_columns.append(
                optional_column
            )

    gt = ground_truth[
        gt_columns
    ].copy()

    gt["true_label"] = 1

    evaluated = scored.merge(
        gt,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    evaluated["true_label"] = (
        evaluated["true_label"]
        .fillna(0)
        .astype(int)
    )

    if evaluated["true_label"].sum() != 30:
        raise ValueError(
            f"{experiment_name}: "
            "ground-truth merge did not produce "
            "exactly 30 anomalies."
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    tp = int(
        (
            (evaluated["rule_prediction"] == 1)
            & (evaluated["true_label"] == 1)
        ).sum()
    )

    fp = int(
        (
            (evaluated["rule_prediction"] == 1)
            & (evaluated["true_label"] == 0)
        ).sum()
    )

    tn = int(
        (
            (evaluated["rule_prediction"] == 0)
            & (evaluated["true_label"] == 0)
        ).sum()
    )

    fn = int(
        (
            (evaluated["rule_prediction"] == 0)
            & (evaluated["true_label"] == 1)
        ).sum()
    )

    accuracy = safe_divide(
        tp + tn,
        len(evaluated),
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    fpr = safe_divide(
        fp,
        fp + tn,
    )

    # --------------------------------------------------------
    # Scenario detection
    # --------------------------------------------------------

    anomalies = evaluated[
        evaluated["true_label"] == 1
    ]

    scenario_rows = []

    for scenario, group in anomalies.groupby(
        "scenario"
    ):

        detected = int(
            group["rule_prediction"].sum()
        )

        total = len(group)

        scenario_rows.append({
            "experiment": experiment_name,
            "difficulty": difficulty,
            "scenario": scenario,
            "anomaly_cases": total,
            "detected": detected,
            "missed": total - detected,
            "detection_rate": safe_divide(
                detected,
                total,
            ),
        })

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    output_path = (
        PREDICTION_DIR
        / f"{experiment_name}_rule_evaluated.csv"
    )

    evaluated.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print(
        f"\n=== {experiment_name} "
        f"({difficulty}) ==="
    )

    print(
        f"Predicted anomalies : "
        f"{int(evaluated['rule_prediction'].sum())}"
    )

    print(
        f"TP={tp} TN={tn} FP={fp} FN={fn}"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"FPR       : {fpr:.4f}"
    )

    print("\nScenario detection:")

    for row in scenario_rows:
        print(
            f"{row['scenario']:25} "
            f"{row['detected']}/"
            f"{row['anomaly_cases']} "
            f"({row['detection_rate']:.4f})"
        )

    return {
        "experiment": experiment_name,
        "difficulty": difficulty,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": fpr,
    }, scenario_rows


def main():

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    thresholds = load_thresholds()

    print(
        "=== Deterministic Rule Baseline ==="
    )

    metric_rows = []
    all_scenario_rows = []

    for experiment, difficulty in (
        EXPERIMENTS.items()
    ):

        metrics, scenario_rows = evaluate(
            experiment,
            difficulty,
            thresholds,
        )

        metric_rows.append(metrics)

        all_scenario_rows.extend(
            scenario_rows
        )

    metrics_df = pd.DataFrame(
        metric_rows
    )

    scenario_df = pd.DataFrame(
        all_scenario_rows
    )

    metrics_path = (
        METRICS_DIR
        / "rule_baseline_metrics.csv"
    )

    scenario_path = (
        METRICS_DIR
        / "rule_baseline_scenario_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    scenario_df.to_csv(
        scenario_path,
        index=False,
    )

    print(
        "\n=== Rule Baseline Comparison ==="
    )

    print(
        metrics_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved metrics: {metrics_path}"
    )

    print(
        f"Saved scenario metrics: "
        f"{scenario_path}"
    )


if __name__ == "__main__":
    main()