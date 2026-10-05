import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCORES_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "normal_validation_scores.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "models"
    / "role_thresholds.json"
)


THRESHOLD_PERCENTILE = 0.95


def main():
    df = pd.read_csv(SCORES_PATH)

    thresholds = {
        "method": (
            "95th percentile of anomaly scores "
            "from independent normal validation data"
        ),
        "percentile": THRESHOLD_PERCENTILE,
        "score_definition": (
            "anomaly_score = "
            "-IsolationForest.score_samples(X); "
            "higher means more unusual"
        ),
        "roles": {},
    }

    print(
        "=== Role-Specific Threshold Calibration ==="
    )

    for role, role_df in df.groupby("role"):

        threshold = float(
            role_df["anomaly_score"].quantile(
                THRESHOLD_PERCENTILE
            )
        )

        flagged = (
            role_df["anomaly_score"]
            > threshold
        ).sum()

        total = len(role_df)

        flagged_rate = flagged / total

        thresholds["roles"][role] = {
            "threshold": threshold,
            "validation_rows": total,
            "validation_rows_above_threshold":
                int(flagged),
            "validation_rate_above_threshold":
                flagged_rate,
        }

        print()
        print(f"{role}")
        print(f"Threshold : {threshold:.6f}")
        print(
            f"Above     : {flagged}/{total} "
            f"({flagged_rate:.2%})"
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
            thresholds,
            file,
            indent=2,
        )

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()