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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "normal_feature_reference.csv"
)


PERCENTILES = [
    0.50,
    0.75,
    0.90,
    0.95,
    0.975,
    0.99,
]


def main():
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    train["dataset"] = "training"
    validation["dataset"] = "validation"

    combined = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    rows = []

    print(
        "=== Normal Feature Reference "
        "(Training + Validation) ==="
    )

    print(
        f"Training rows   : {len(train)}"
    )

    print(
        f"Validation rows : {len(validation)}"
    )

    print(
        f"Combined rows   : {len(combined)}"
    )

    for role in sorted(
        combined["role"].unique()
    ):

        role_df = combined[
            combined["role"] == role
        ]

        print()
        print("=" * 90)
        print(f"ROLE: {role}")
        print(
            f"Normal reference observations: "
            f"{len(role_df)}"
        )
        print("=" * 90)

        role_rows = []

        for feature in ML_FEATURES:

            values = role_df[feature]

            result = {
                "role": role,
                "feature": feature,
                "count": len(values),
                "min": values.min(),
                "mean": values.mean(),
                "std": values.std(),
                "p01": values.quantile(0.01),
                "p02_5": values.quantile(0.025),
                "p05": values.quantile(0.05),
                "p10": values.quantile(0.10),                                
                "p50": values.quantile(0.50),
                "p75": values.quantile(0.75),
                "p90": values.quantile(0.90),
                "p95": values.quantile(0.95),
                "p97_5": values.quantile(0.975),
                "p99": values.quantile(0.99),
                "max": values.max(),
            }

            rows.append(result)
            role_rows.append(result)

        role_summary = pd.DataFrame(
            role_rows
        )

        display_columns = [
            "feature",
            "min",
            "mean",
            "std",
            "p01",
            "p02_5",
            "p05",
            "p10",            
            "p50",
            "p75",
            "p90",
            "p95",
            "p97_5",
            "p99",
            "max",
        ]

        print(
            role_summary[
                display_columns
            ]
            .round(3)
            .to_string(index=False)
        )

    reference_df = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Saved reference table: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()