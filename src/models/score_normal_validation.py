from pathlib import Path

import joblib
import pandas as pd

from src.features.feature_config import ML_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "normal_validation_user_day.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "normal_validation_scores.csv"
)


ROLE_MODEL_FILES = {
    "Teller": "isolation_forest_teller.joblib",
    "Customer Service":
        "isolation_forest_customer_service.joblib",
    "Manager": "isolation_forest_manager.joblib",
}


def main():
    df = pd.read_csv(VALIDATION_PATH)

    scored_frames = []

    print("=== Normal Validation Scoring ===")
    print(f"Validation rows: {len(df)}")
    print()

    for role, model_filename in ROLE_MODEL_FILES.items():

        role_df = df[
            df["role"] == role
        ].copy()

        model_path = MODEL_DIR / model_filename

        model_package = joblib.load(model_path)

        model = model_package["model"]
        model_features = model_package["features"]

        if model_features != ML_FEATURES:
            raise ValueError(
                f"Feature mismatch for {role} model."
            )

        X = role_df[ML_FEATURES]

        raw_score = model.score_samples(X)

        role_df["iforest_raw_score"] = raw_score

        role_df["anomaly_score"] = -raw_score

        scored_frames.append(role_df)

        print(f"=== {role} ===")
        print(f"Rows: {len(role_df)}")
        print(
            "Anomaly score min : "
            f"{role_df['anomaly_score'].min():.6f}"
        )
        print(
            "Anomaly score mean: "
            f"{role_df['anomaly_score'].mean():.6f}"
        )
        print(
            "Anomaly score max : "
            f"{role_df['anomaly_score'].max():.6f}"
        )
        print()

    scored_df = pd.concat(
        scored_frames,
        ignore_index=True,
    )

    scored_df = scored_df.sort_values(
        ["role", "date", "user_id"]
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scored_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()