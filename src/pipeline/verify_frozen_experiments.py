from pathlib import Path

import pandas as pd

from src.pipeline.detection_pipeline import (
    run_feature_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
)


EXPECTED = {
    "experiment_1": {
        "ml_anomalies": 27,
        "rule_anomalies": 72,
        "candidates": 76,
    },
    "experiment_2": {
        "ml_anomalies": 27,
        "rule_anomalies": 72,
        "candidates": 76,
    },
    "experiment_3": {
        "ml_anomalies": 21,
        "rule_anomalies": 68,
        "candidates": 72,
    },
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


def verify_ml_identity(
    experiment,
    runtime_ml,
):
    frozen_path = (
        PREDICTION_DIR
        / f"{experiment}_predictions.csv"
    )

    frozen = normalize_keys(
        pd.read_csv(frozen_path)
    )

    runtime = normalize_keys(
        runtime_ml
    )

    compare = frozen[
        KEY_COLUMNS
        + [
            "anomaly_score",
            "threshold",
            "prediction",
        ]
    ].merge(
        runtime[
            KEY_COLUMNS
            + [
                "anomaly_score",
                "threshold",
                "prediction",
            ]
        ],
        on=KEY_COLUMNS,
        how="outer",
        suffixes=(
            "_frozen",
            "_runtime",
        ),
        indicator=True,
        validate="one_to_one",
    )

    if not (
        compare["_merge"] == "both"
    ).all():
        raise ValueError(
            f"{experiment}: runtime/frozen "
            "ML keys differ."
        )

    prediction_match = (
        compare["prediction_frozen"]
        .astype(int)
        ==
        compare["prediction_runtime"]
        .astype(int)
    )

    if not prediction_match.all():
        raise ValueError(
            f"{experiment}: ML predictions "
            "do not reproduce frozen results."
        )

    score_difference = (
        compare["anomaly_score_frozen"]
        - compare["anomaly_score_runtime"]
    ).abs()

    if score_difference.max() > 1e-10:
        raise ValueError(
            f"{experiment}: ML anomaly scores "
            "do not reproduce frozen results."
        )


def verify_rule_identity(
    experiment,
    runtime_rules,
):
    frozen_path = (
        PREDICTION_DIR
        / f"{experiment}_rule_evaluated.csv"
    )

    frozen = normalize_keys(
        pd.read_csv(frozen_path)
    )

    runtime = normalize_keys(
        runtime_rules
    )

    columns = [
        "rule_after_hours",
        "rule_large_export",
        "rule_high_record_access",
        "rules_triggered",
        "rule_prediction",
    ]

    compare = frozen[
        KEY_COLUMNS + columns
    ].merge(
        runtime[
            KEY_COLUMNS + columns
        ],
        on=KEY_COLUMNS,
        how="outer",
        suffixes=(
            "_frozen",
            "_runtime",
        ),
        indicator=True,
        validate="one_to_one",
    )

    if not (
        compare["_merge"] == "both"
    ).all():
        raise ValueError(
            f"{experiment}: runtime/frozen "
            "rule keys differ."
        )

    for column in columns:

        matches = (
            compare[
                f"{column}_frozen"
            ].astype(int)
            ==
            compare[
                f"{column}_runtime"
            ].astype(int)
        )

        if not matches.all():
            raise ValueError(
                f"{experiment}: rule column "
                f"{column} does not reproduce "
                "frozen results."
            )


def verify_experiment(experiment):
    feature_path = (
        PROCESSED_DIR
        / f"{experiment}_user_day.csv"
    )

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Missing feature file: "
            f"{feature_path}"
        )

    features = pd.read_csv(
        feature_path
    )

    result = run_feature_pipeline(
        features
    )

    combined = result["combined"]

    actual = {
        "ml_anomalies": int(
            combined["prediction"].sum()
        ),
        "rule_anomalies": int(
            combined[
                "rule_prediction"
            ].sum()
        ),
        "candidates": int(
            combined["candidate"].sum()
        ),
    }

    expected = EXPECTED[
        experiment
    ]

    print(
        f"\n=== {experiment} ==="
    )

    print(
        f"User-days       : "
        f"{len(combined)}"
    )

    print(
        f"ML anomalies    : "
        f"{actual['ml_anomalies']} "
        f"(expected "
        f"{expected['ml_anomalies']})"
    )

    print(
        f"Rule anomalies  : "
        f"{actual['rule_anomalies']} "
        f"(expected "
        f"{expected['rule_anomalies']})"
    )

    print(
        f"Candidates      : "
        f"{actual['candidates']} "
        f"(expected "
        f"{expected['candidates']})"
    )

    if actual != expected:
        raise ValueError(
            f"{experiment}: detector counts "
            "do not reproduce frozen results."
        )

    verify_ml_identity(
        experiment,
        result["ml_results"],
    )

    verify_rule_identity(
        experiment,
        result["rule_results"],
    )

    print(
        "Frozen ML identity   : PASS"
    )

    print(
        "Frozen rule identity : PASS"
    )

    print(
        "Candidate count      : PASS"
    )


def main():
    print(
        "=== Stage 26 Frozen Experiment "
        "Reproduction Test ==="
    )

    for experiment in [
        "experiment_1",
        "experiment_2",
        "experiment_3",
    ]:
        verify_experiment(
            experiment
        )

    print(
        "\n================================"
    )

    print(
        "STAGE 26 REPRODUCTION: PASS"
    )

    print(
        "================================"
    )

    print(
        "\nNo model retraining performed."
    )

    print(
        "No threshold calibration performed."
    )

    print(
        "No ground truth used."
    )

    print(
        "No OpenAI API call performed."
    )


if __name__ == "__main__":
    main()