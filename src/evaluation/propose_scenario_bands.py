from pathlib import Path

import pandas as pd


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
    / "scenario_band_reference.csv"
)


def main():
    reference = pd.read_csv(
        REFERENCE_PATH
    )

    rows = []

    print(
        "=== Candidate Scenario Band Reference ==="
    )

    print(
        "\nIMPORTANT:"
        "\nThese values are design references only."
        "\nThey do not automatically define anomalies."
        "\nDifficulty levels must be reviewed in "
        "behavioural context before test generation."
    )

    for _, row in reference.iterrows():

        mean = row["mean"]
        std = row["std"]

        result = {
            "role": row["role"],
            "feature": row["feature"],

            "normal_mean": mean,
            "normal_std": std,

            "p90": row["p90"],
            "p95": row["p95"],
            "p97_5": row["p97_5"],
            "p99": row["p99"],
            "normal_max": row["max"],

            "mean_plus_1sd":
                mean + std,

            "mean_plus_2sd":
                mean + (2 * std),

            "mean_plus_3sd":
                mean + (3 * std),
        }

        rows.append(result)

    bands = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    bands.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    for role in sorted(
        bands["role"].unique()
    ):

        print()
        print("=" * 100)
        print(f"ROLE: {role}")
        print("=" * 100)

        role_df = bands[
            bands["role"] == role
        ]

        columns = [
            "feature",
            "normal_mean",
            "normal_std",
            "p90",
            "p95",
            "p97_5",
            "p99",
            "normal_max",
            "mean_plus_1sd",
            "mean_plus_2sd",
            "mean_plus_3sd",
        ]

        print(
            role_df[
                columns
            ]
            .round(3)
            .to_string(index=False)
        )

    print()
    print(
        f"Saved scenario reference: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()