from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

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


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0

    return numerator / denominator


def load_experiment(experiment_name, difficulty):
    path = (
        PREDICTION_DIR
        / f"{experiment_name}_evaluated.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluated file not found: {path}"
        )

    df = pd.read_csv(path)

    required = [
        "user_id",
        "date",
        "role",
        "anomaly_score",
        "threshold",
        "prediction",
        "true_label",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{experiment_name}: "
            f"missing columns {missing}"
        )

    df["experiment"] = experiment_name
    df["experiment_difficulty"] = difficulty

    return df


def calculate_role_metrics(df):
    rows = []

    for role, group in df.groupby("role"):

        tp = int(
            (
                (group["prediction"] == 1)
                & (group["true_label"] == 1)
            ).sum()
        )

        tn = int(
            (
                (group["prediction"] == 0)
                & (group["true_label"] == 0)
            ).sum()
        )

        fp = int(
            (
                (group["prediction"] == 1)
                & (group["true_label"] == 0)
            ).sum()
        )

        fn = int(
            (
                (group["prediction"] == 0)
                & (group["true_label"] == 1)
            ).sum()
        )

        precision = safe_divide(
            tp,
            tp + fp,
        )

        recall = safe_divide(
            tp,
            tp + fn,
        )

        fpr = safe_divide(
            fp,
            fp + tn,
        )

        rows.append({
            "role": role,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "false_positive_rate": fpr,
        })

    return pd.DataFrame(rows)


def calculate_scenario_detection(df):
    anomalies = df[
        df["true_label"] == 1
    ].copy()

    if "scenario" not in anomalies.columns:
        raise ValueError(
            "Scenario column not found in evaluated data."
        )

    rows = []

    for scenario, group in anomalies.groupby(
        "scenario"
    ):

        total = len(group)

        detected = int(
            (group["prediction"] == 1).sum()
        )

        missed = int(
            (group["prediction"] == 0).sum()
        )

        detection_rate = safe_divide(
            detected,
            total,
        )

        rows.append({
            "scenario": scenario,
            "anomaly_cases": total,
            "detected": detected,
            "missed": missed,
            "detection_rate": detection_rate,
        })

    return pd.DataFrame(rows)


def build_error_cases(df):
    errors = df[
        df["prediction"]
        != df["true_label"]
    ].copy()

    errors["error_type"] = "UNKNOWN"

    errors.loc[
        (
            (errors["prediction"] == 1)
            & (errors["true_label"] == 0)
        ),
        "error_type",
    ] = "FP"

    errors.loc[
        (
            (errors["prediction"] == 0)
            & (errors["true_label"] == 1)
        ),
        "error_type",
    ] = "FN"

    errors["score_margin"] = (
        errors["anomaly_score"]
        - errors["threshold"]
    )

    preferred_columns = [
        "experiment",
        "experiment_difficulty",
        "case_id",
        "user_id",
        "date",
        "role",
        "scenario",
        "error_type",
        "anomaly_score",
        "threshold",
        "score_margin",
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in errors.columns
    ]

    return errors[
        available_columns
    ].sort_values(
        [
            "error_type",
            "role",
            "anomaly_score",
        ]
    )


def compare_shared_anomaly_cases(all_data):
    """
    Compare the same injected anomaly identities across
    obvious, moderate and subtle experiments.
    """

    anomaly_frames = []

    for df in all_data:

        anomaly_df = df[
            df["true_label"] == 1
        ].copy()

        columns = (
            KEY_COLUMNS
            + [
                "experiment",
                "experiment_difficulty",
                "anomaly_score",
                "threshold",
                "prediction",
            ]
        )

        if "case_id" in anomaly_df.columns:
            columns.insert(0, "case_id")

        if "scenario" in anomaly_df.columns:
            columns.append("scenario")

        anomaly_frames.append(
            anomaly_df[columns]
        )

    combined = pd.concat(
        anomaly_frames,
        ignore_index=True,
    )

    return combined.sort_values(
        [
            "role",
            "user_id",
            "date",
            "experiment",
        ]
    )


def main():
    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=== Experiment Diagnostic Analysis ==="
    )

    all_data = []
    role_results = []
    scenario_results = []
    error_results = []

    for experiment_name, difficulty in (
        EXPERIMENTS.items()
    ):

        df = load_experiment(
            experiment_name,
            difficulty,
        )

        all_data.append(df)

        # ----------------------------------------------------
        # Role analysis
        # ----------------------------------------------------

        role_df = calculate_role_metrics(df)

        role_df.insert(
            0,
            "difficulty",
            difficulty,
        )

        role_df.insert(
            0,
            "experiment",
            experiment_name,
        )

        role_results.append(role_df)

        # ----------------------------------------------------
        # Scenario analysis
        # ----------------------------------------------------

        scenario_df = (
            calculate_scenario_detection(df)
        )

        scenario_df.insert(
            0,
            "difficulty",
            difficulty,
        )

        scenario_df.insert(
            0,
            "experiment",
            experiment_name,
        )

        scenario_results.append(
            scenario_df
        )

        # ----------------------------------------------------
        # FP / FN cases
        # ----------------------------------------------------

        error_df = build_error_cases(df)

        error_results.append(error_df)

    # ========================================================
    # Combine results
    # ========================================================

    role_metrics = pd.concat(
        role_results,
        ignore_index=True,
    )

    scenario_metrics = pd.concat(
        scenario_results,
        ignore_index=True,
    )

    error_cases = pd.concat(
        error_results,
        ignore_index=True,
    )

    case_comparison = (
        compare_shared_anomaly_cases(
            all_data
        )
    )

    # ========================================================
    # Save
    # ========================================================

    role_path = (
        METRICS_DIR
        / "experiment_role_metrics.csv"
    )

    scenario_path = (
        METRICS_DIR
        / "experiment_scenario_metrics.csv"
    )

    error_path = (
        METRICS_DIR
        / "experiment_error_cases.csv"
    )

    comparison_path = (
        METRICS_DIR
        / "experiment_anomaly_case_comparison.csv"
    )

    role_metrics.to_csv(
        role_path,
        index=False,
    )

    scenario_metrics.to_csv(
        scenario_path,
        index=False,
    )

    error_cases.to_csv(
        error_path,
        index=False,
    )

    case_comparison.to_csv(
        comparison_path,
        index=False,
    )

    # ========================================================
    # Console summaries
    # ========================================================

    print(
        "\n=== Detection by Role ==="
    )

    print(
        role_metrics[
            [
                "experiment",
                "difficulty",
                "role",
                "tp",
                "fp",
                "fn",
                "recall",
                "false_positive_rate",
            ]
        ].to_string(index=False)
    )

    print(
        "\n=== Detection by Scenario ==="
    )

    print(
        scenario_metrics[
            [
                "experiment",
                "difficulty",
                "scenario",
                "anomaly_cases",
                "detected",
                "missed",
                "detection_rate",
            ]
        ].to_string(index=False)
    )

    print(
        "\n=== Errors ==="
    )

    error_summary = (
        error_cases
        .groupby(
            [
                "experiment",
                "experiment_difficulty",
                "error_type",
            ]
        )
        .size()
        .reset_index(name="count")
    )

    print(
        error_summary.to_string(
            index=False
        )
    )

    print(
        "\n=== Files Saved ==="
    )

    print(role_path)
    print(scenario_path)
    print(error_path)
    print(comparison_path)


if __name__ == "__main__":
    main()