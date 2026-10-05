from pathlib import Path

import pandas as pd

from src.features.feature_config import ML_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "normal_feature_reference.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "role_normality_comparison.csv"
)


def main():
    df = pd.read_csv(
        REFERENCE_PATH
    )

    comparison_rows = []

    print(
        "=== Role-Specific Normal Behaviour Comparison ==="
    )

    for feature in ML_FEATURES:

        feature_df = df[
            df["feature"] == feature
        ]

        print()
        print("=" * 90)
        print(f"FEATURE: {feature}")
        print("=" * 90)

        display = feature_df[
            [
                "role",
                "mean",
                "std",
                "p05",
                "p50",
                "p95",
                "max",
            ]
        ].copy()

        print(
            display
            .round(3)
            .to_string(index=False)
        )

        for _, row in display.iterrows():
            comparison_rows.append(
                {
                    "feature": feature,
                    "role": row["role"],
                    "mean": row["mean"],
                    "std": row["std"],
                    "p05": row["p05"],
                    "p50": row["p50"],
                    "p95": row["p95"],
                    "max": row["max"],
                }
            )

    output_df = pd.DataFrame(
        comparison_rows
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Saved role comparison: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()