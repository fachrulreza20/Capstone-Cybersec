from pathlib import Path

import pandas as pd

from src.features.feature_config import ML_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "normal_training_user_day.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "normal_validation_user_day.csv"
)


def main():
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    print(
        "=== Training vs Validation Feature Means ==="
    )

    for role in sorted(train["role"].unique()):

        print()
        print(f"=== {role} ===")

        train_role = train[
            train["role"] == role
        ]

        validation_role = validation[
            validation["role"] == role
        ]

        rows = []

        for feature in ML_FEATURES:

            train_mean = train_role[
                feature
            ].mean()

            validation_mean = validation_role[
                feature
            ].mean()

            train_std = train_role[
                feature
            ].std()

            validation_std = validation_role[
                feature
            ].std()

            rows.append(
                {
                    "feature": feature,
                    "train_mean": train_mean,
                    "validation_mean":
                        validation_mean,
                    "train_std": train_std,
                    "validation_std":
                        validation_std,
                }
            )

        comparison = pd.DataFrame(rows)

        print(
            comparison.round(3).to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()