from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCORES_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "normal_validation_scores.csv"
)


def main():
    df = pd.read_csv(SCORES_PATH)

    print(
        "=== Validation Anomaly Score Distribution ==="
    )

    for role, role_df in df.groupby("role"):

        scores = role_df["anomaly_score"]

        print()
        print(f"=== {role} ===")

        print(f"Count : {len(scores)}")
        print(f"Min   : {scores.min():.6f}")
        print(f"Mean  : {scores.mean():.6f}")
        print(f"Std   : {scores.std():.6f}")
        print(f"Max   : {scores.max():.6f}")

        print("\nPercentiles:")

        for percentile in [
            0.50,
            0.90,
            0.95,
            0.975,
            0.99,
        ]:

            value = scores.quantile(percentile)

            print(
                f"{percentile * 100:5.1f}% : "
                f"{value:.6f}"
            )

        print("\nTop 5 highest anomaly scores:")

        columns = [
            "user_id",
            "date",
            "anomaly_score",
        ]

        print(
            role_df.nlargest(
                5,
                "anomaly_score",
            )[columns].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()