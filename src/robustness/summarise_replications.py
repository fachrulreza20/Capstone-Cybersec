from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "robustness"
    / "replication_metrics.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "robustness"
    / "replication_summary.csv"
)

TABLE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "robustness"
    / "final_performance_table.csv"
)


METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "fpr",
    "specificity",
    "flagged",
]


CONDITION_ORDER = [
    "control",
    "obvious",
    "moderate",
    "subtle",
]

DETECTOR_ORDER = [
    "ml",
    "rules",
]


def confidence_interval_95(values):
    """
    95% CI for the mean across independent replications.

    Uses:
        mean +/- 1.96 * standard_error

    With n=50 independent replications this normal
    approximation is appropriate for the descriptive
    robustness analysis used in this project.
    """

    values = pd.Series(values).dropna()

    n = len(values)

    if n == 0:
        return np.nan, np.nan

    if n == 1:
        value = float(values.iloc[0])
        return value, value

    mean = float(values.mean())

    std = float(
        values.std(ddof=1)
    )

    standard_error = std / np.sqrt(n)

    # margin = 1.96 * standard_error


    T_CRITICAL_95_DF49 = 2.009575
    margin = T_CRITICAL_95_DF49 * standard_error


    return (
        mean - margin,
        mean + margin,
    )


def validate_input(df):
    required_columns = {
        "condition",
        "replication",
        "detector",
        "n_user_days",
        "n_anomalies",
        "tp",
        "tn",
        "fp",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fpr",
        "specificity",
        "flagged",
    }

    missing = sorted(
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    expected_conditions = set(
        CONDITION_ORDER
    )

    actual_conditions = set(
        df["condition"].unique()
    )

    if actual_conditions != expected_conditions:
        raise ValueError(
            "Unexpected condition set.\n"
            f"Expected: {expected_conditions}\n"
            f"Actual: {actual_conditions}"
        )

    for condition in CONDITION_ORDER:
        for detector in DETECTOR_ORDER:
            subset = df[
                (df["condition"] == condition)
                & (df["detector"] == detector)
            ]

            if len(subset) != 50:
                raise ValueError(
                    f"{condition}/{detector}: "
                    f"expected 50 rows, "
                    f"found {len(subset)}."
                )

            if (
                subset["replication"].nunique()
                != 50
            ):
                raise ValueError(
                    f"{condition}/{detector}: "
                    "replication IDs are not unique."
                )

            if not (
                subset["n_user_days"] == 300
            ).all():
                raise ValueError(
                    f"{condition}/{detector}: "
                    "not all runs contain "
                    "300 user-days."
                )

            expected_anomalies = (
                0
                if condition == "control"
                else 30
            )

            if not (
                subset["n_anomalies"]
                == expected_anomalies
            ).all():
                raise ValueError(
                    f"{condition}/{detector}: "
                    "unexpected anomaly count."
                )


def build_summary(df):
    rows = []

    for condition in CONDITION_ORDER:
        for detector in DETECTOR_ORDER:

            subset = df[
                (df["condition"] == condition)
                & (df["detector"] == detector)
            ].copy()

            for metric in METRICS:

                values = (
                    subset[metric]
                    .replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )
                    .dropna()
                )

                n = len(values)

                if n == 0:
                    continue

                ci_lower, ci_upper = (
                    confidence_interval_95(
                        values
                    )
                )

                rows.append(
                    {
                        "condition": condition,
                        "detector": detector,
                        "metric": metric,
                        "n_replications": n,
                        "mean": values.mean(),
                        "std": (
                            values.std(ddof=1)
                            if n > 1
                            else 0.0
                        ),
                        "ci95_lower": ci_lower,
                        "ci95_upper": ci_upper,
                        "min": values.min(),
                        "max": values.max(),
                    }
                )

    return pd.DataFrame(rows)


def build_final_table(summary):
    """
    Create compact report-facing table.

    Control:
        Accuracy, FPR, Specificity

    Anomaly conditions:
        Accuracy, Precision, Recall, F1, FPR
    """

    rows = []

    for condition in CONDITION_ORDER:
        for detector in DETECTOR_ORDER:

            if condition == "control":
                selected_metrics = [
                    "accuracy",
                    "fpr",
                    "specificity",
                ]
            else:
                selected_metrics = [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "fpr",
                ]

            for metric in selected_metrics:

                match = summary[
                    (
                        summary["condition"]
                        == condition
                    )
                    & (
                        summary["detector"]
                        == detector
                    )
                    & (
                        summary["metric"]
                        == metric
                    )
                ]

                if len(match) != 1:
                    raise ValueError(
                        "Expected exactly one "
                        f"summary row for "
                        f"{condition}/{detector}/{metric}."
                    )

                row = match.iloc[0]

                rows.append(
                    {
                        "condition": condition,
                        "detector": detector,
                        "metric": metric,
                        "mean": row["mean"],
                        "std": row["std"],
                        "ci95_lower": row[
                            "ci95_lower"
                        ],
                        "ci95_upper": row[
                            "ci95_upper"
                        ],
                        "report_value": (
                            f"{row['mean'] * 100:.2f}% "
                            f"(95% CI "
                            f"{row['ci95_lower'] * 100:.2f}"
                            f"–"
                            f"{row['ci95_upper'] * 100:.2f}%)"
                        ),
                    }
                )

    return pd.DataFrame(rows)


def print_report_table(final_table):
    print(
        "\n=== FINAL REPLICATED PERFORMANCE ==="
    )

    for condition in CONDITION_ORDER:

        print(
            f"\n--- {condition.upper()} ---"
        )

        for detector in DETECTOR_ORDER:

            print(
                f"\n{detector.upper()}"
            )

            subset = final_table[
                (
                    final_table["condition"]
                    == condition
                )
                & (
                    final_table["detector"]
                    == detector
                )
            ]

            for _, row in subset.iterrows():

                metric_name = (
                    row["metric"]
                    .replace("_", " ")
                    .upper()
                )

                print(
                    f"{metric_name:12s}: "
                    f"{row['report_value']}"
                )


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    print(
        "=== Stage 30D.2 Statistical Aggregation ==="
    )

    print(
        f"Input rows       : {len(df)}"
    )

    print(
        f"Datasets         : "
        f"{df[['condition', 'replication']].drop_duplicates().shape[0]}"
    )

    print(
        f"Metric rows      : {len(df)}"
    )

    validate_input(df)

    print(
        "Input validation : PASS"
    )

    summary = build_summary(df)

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    final_table = build_final_table(
        summary
    )

    final_table.to_csv(
        TABLE_PATH,
        index=False,
    )

    print_report_table(
        final_table
    )

    print(
        "\n=== OUTPUTS ==="
    )

    print(
        f"Full summary : {SUMMARY_PATH}"
    )

    print(
        f"Report table : {TABLE_PATH}"
    )

    print(
        "\nSTAGE 30D.2: PASS"
    )


if __name__ == "__main__":
    main()