import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCORES_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "normal_validation_scores.csv"
)

THRESHOLD_PATH = (
    PROJECT_ROOT
    / "models"
    / "role_thresholds.json"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
)


def safe_role_name(role):
    return (
        role.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def main():
    df = pd.read_csv(SCORES_PATH)

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        threshold_data = json.load(file)

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for role, role_df in df.groupby("role"):

        threshold = (
            threshold_data["roles"][role][
                "threshold"
            ]
        )

        plt.figure(figsize=(8, 5))

        plt.hist(
            role_df["anomaly_score"],
            bins=20,
        )

        plt.axvline(
            threshold,
            linestyle="--",
            label=(
                "95th percentile threshold "
                f"({threshold:.3f})"
            ),
        )

        plt.xlabel("Anomaly Score")
        plt.ylabel("Number of Normal Validation Days")

        plt.title(
            f"{role} — Normal Validation "
            "Anomaly Score Distribution"
        )

        plt.legend()
        plt.tight_layout()

        output_path = (
            FIGURE_DIR
            / (
                "validation_score_"
                f"{safe_role_name(role)}.png"
            )
        )

        plt.savefig(
            output_path,
            dpi=200,
        )

        plt.close()

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()