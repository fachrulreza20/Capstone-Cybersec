import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.features.feature_config import ML_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "normal_training_user_day.csv"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "project_config.json"
)

MODEL_DIR = PROJECT_ROOT / "models"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def safe_role_name(role):
    return (
        role.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def main():
    config = load_config()

    df = pd.read_csv(DATA_PATH)

    seed = config["random_seed"]
    model_config = config["isolation_forest"]

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=== Role-Specific Isolation Forest Training ===")
    print(f"Training rows : {len(df)}")
    print(f"Features      : {len(ML_FEATURES)}")
    print()

    print("ML features:")

    for feature in ML_FEATURES:
        print(f"  - {feature}")

    print()

    for role in config["roles"]:

        role_df = df[
            df["role"] == role
        ].copy()

        X_train = role_df[ML_FEATURES]

        print(f"=== Training: {role} ===")
        print(f"Rows: {len(X_train)}")

        model = IsolationForest(
            n_estimators=model_config["n_estimators"],
            max_samples=model_config["max_samples"],
            contamination=model_config["contamination"],
            random_state=seed,
            n_jobs=-1,
        )

        model.fit(X_train)

        model_filename = (
            f"isolation_forest_"
            f"{safe_role_name(role)}.joblib"
        )

        model_path = (
            MODEL_DIR
            / model_filename
        )

        model_package = {
            "role": role,
            "features": ML_FEATURES,
            "model": model,
            "training_rows": len(X_train),
            "random_seed": seed,
        }

        joblib.dump(
            model_package,
            model_path,
        )

        print(f"Saved: {model_path}")
        print()

    print(
        "All role-specific models trained successfully."
    )


if __name__ == "__main__":
    main()