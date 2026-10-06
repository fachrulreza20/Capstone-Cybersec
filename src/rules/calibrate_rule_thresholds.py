import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "normal_validation_user_day.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "models"
    / "rule_thresholds.json"
)

PERCENTILE = 0.95

RULE_FEATURES = {
    "after_hours": "last_activity_hour",
    "large_export": "download_total",
    "high_record_access": "records_accessed_total",
}


def main():

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation feature file not found: "
            f"{VALIDATION_PATH}"
        )

    df = pd.read_csv(VALIDATION_PATH)

    required_columns = [
        "role",
        *RULE_FEATURES.values(),
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing validation columns: {missing}"
        )

    if df[required_columns].isna().any().any():
        raise ValueError(
            "NaN found in validation rule features."
        )

    output = {
        "method": (
            "95th percentile of independent normal "
            "validation behaviour per role"
        ),
        "percentile": PERCENTILE,
        "classification_rule": (
            "prediction = anomaly when one or more "
            "rules are triggered"
        ),
        "roles": {},
    }

    print("=== Rule Threshold Calibration ===")

    for role in sorted(df["role"].unique()):

        role_df = df[
            df["role"] == role
        ]

        role_thresholds = {}

        print(f"\n{role}")

        for rule_name, feature in (
            RULE_FEATURES.items()
        ):

            threshold = float(
                role_df[feature].quantile(
                    PERCENTILE
                )
            )

            role_thresholds[
                rule_name
            ] = {
                "feature": feature,
                "threshold": threshold,
            }

            print(
                f"{rule_name:22} "
                f"{feature:28} "
                f"{threshold:.4f}"
            )

        output["roles"][role] = (
            role_thresholds
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()