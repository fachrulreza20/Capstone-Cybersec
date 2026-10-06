from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
)

GROUND_TRUTH_DIR = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
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
    """
    Normalize identity columns before merging.
    """

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
    """
    Prevent division-by-zero errors.
    """

    if denominator == 0:
        return 0.0

    return numerator / denominator


def evaluate_experiment(
    experiment_name,
    difficulty,
):
    prediction_path = (
        PREDICTION_DIR
        / f"{experiment_name}_predictions.csv"
    )

    ground_truth_path = (
        GROUND_TRUTH_DIR
        / f"{experiment_name}_ground_truth.csv"
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: "
            f"{prediction_path}"
        )

    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found: "
            f"{ground_truth_path}"
        )

    predictions = pd.read_csv(
        prediction_path
    )

    ground_truth = pd.read_csv(
        ground_truth_path
    )

    predictions = normalize_keys(
        predictions
    )

    ground_truth = normalize_keys(
        ground_truth
    )

    # --------------------------------------------------------
    # Integrity checks
    # --------------------------------------------------------

    if len(predictions) != 300:
        raise ValueError(
            f"{experiment_name}: "
            f"expected 300 predictions, "
            f"found {len(predictions)}."
        )

    if len(ground_truth) != 30:
        raise ValueError(
            f"{experiment_name}: "
            f"expected 30 anomaly ground-truth rows, "
            f"found {len(ground_truth)}."
        )

    if predictions.duplicated(
        KEY_COLUMNS
    ).any():
        raise ValueError(
            f"{experiment_name}: "
            "duplicate prediction identities found."
        )

    if ground_truth.duplicated(
        KEY_COLUMNS
    ).any():
        raise ValueError(
            f"{experiment_name}: "
            "duplicate ground-truth identities found."
        )

    # --------------------------------------------------------
    # Ground truth contains only the 30 injected anomalies.
    #
    # Therefore:
    # matched row   -> true_label = 1
    # unmatched row -> true_label = 0
    # --------------------------------------------------------

    gt_metadata_columns = [
        column
        for column in [
            "case_id",
            "scenario",
            "difficulty",
        ]
        if column in ground_truth.columns
    ]

    gt_for_merge = ground_truth[
        KEY_COLUMNS
        + gt_metadata_columns
    ].copy()

    gt_for_merge["true_label"] = 1

    evaluated = predictions.merge(
        gt_for_merge,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    evaluated["true_label"] = (
        evaluated["true_label"]
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Verify exactly 30 injected anomalies matched
    # --------------------------------------------------------

    positive_count = int(
        evaluated["true_label"].sum()
    )

    if positive_count != 30:
        raise ValueError(
            f"{experiment_name}: "
            f"expected 30 matched anomalies, "
            f"found {positive_count}."
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    tp = int(
        (
            (evaluated["prediction"] == 1)
            & (evaluated["true_label"] == 1)
        ).sum()
    )

    fp = int(
        (
            (evaluated["prediction"] == 1)
            & (evaluated["true_label"] == 0)
        ).sum()
    )

    tn = int(
        (
            (evaluated["prediction"] == 0)
            & (evaluated["true_label"] == 0)
        ).sum()
    )

    fn = int(
        (
            (evaluated["prediction"] == 0)
            & (evaluated["true_label"] == 1)
        ).sum()
    )

    total = len(evaluated)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = safe_divide(
        tp + tn,
        total,
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

    false_positive_rate = safe_divide(
        fp,
        fp + tn,
    )

    # --------------------------------------------------------
    # Save evaluated rows
    # --------------------------------------------------------

    output_path = (
        PREDICTION_DIR
        / f"{experiment_name}_evaluated.csv"
    )

    evaluated.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print(
        f"\n=== {experiment_name} "
        f"({difficulty}) ==="
    )

    print(
        f"Total user-days : {total}"
    )

    print(
        f"True anomalies  : "
        f"{positive_count}"
    )

    print()
    print("Confusion Matrix Counts")
    print(f"TP : {tp}")
    print(f"TN : {tn}")
    print(f"FP : {fp}")
    print(f"FN : {fn}")

    print()
    print("Performance Metrics")

    print(
        f"Accuracy  : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision : "
        f"{precision:.4f}"
    )

    print(
        f"Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{f1:.4f}"
    )

    print(
        f"FPR       : "
        f"{false_positive_rate:.4f}"
    )

    print(
        f"\nSaved evaluated predictions: "
        f"{output_path}"
    )

    return {
        "experiment": experiment_name,
        "difficulty": difficulty,
        "total_user_days": total,
        "true_anomalies": positive_count,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": false_positive_rate,
    }


def main():

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=== Experiment Evaluation ==="
    )

    results = []

    for (
        experiment_name,
        difficulty,
    ) in EXPERIMENTS.items():

        result = evaluate_experiment(
            experiment_name,
            difficulty,
        )

        results.append(result)

    # --------------------------------------------------------
    # Save summary metrics
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        results
    )

    metrics_path = (
        METRICS_DIR
        / "experiment_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    print(
        "\n=== Experiment Comparison ==="
    )

    display_columns = [
        "experiment",
        "difficulty",
        "tp",
        "tn",
        "fp",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "false_positive_rate",
    ]

    print(
        metrics_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        f"\nSaved summary metrics: "
        f"{metrics_path}"
    )


if __name__ == "__main__":
    main()