import json
from pathlib import Path

import joblib
import pandas as pd

from src.features.feature_config import ML_FEATURES


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiment_config.json"
)

THRESHOLD_PATH = (
    PROJECT_ROOT
    / "models"
    / "role_thresholds.json"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
)


# ============================================================
# LOAD CONFIGURATION
# ============================================================

def load_config():
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# LOAD ROLE-SPECIFIC THRESHOLDS
# ============================================================

def load_thresholds():
    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if "roles" not in data:
        raise ValueError(
            "Invalid threshold file: "
            "'roles' section not found."
        )

    thresholds = {}

    for role, role_info in data["roles"].items():

        if "threshold" not in role_info:
            raise ValueError(
                f"Threshold missing for role: {role}"
            )

        thresholds[role] = float(
            role_info["threshold"]
        )

    return thresholds


# ============================================================
# MODEL FILE NAME
# ============================================================

def model_filename(role):
    safe_role = (
        role
        .lower()
        .replace(" ", "_")
    )

    return (
        MODEL_DIR
        / f"isolation_forest_{safe_role}.joblib"
    )


# ============================================================
# SCORE ONE EXPERIMENT
# ============================================================

def score_experiment(
    experiment_name,
    thresholds,
):

    input_path = (
        PROCESSED_DIR
        / f"{experiment_name}_user_day.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Experiment feature file not found: "
            f"{input_path}"
        )

    df = pd.read_csv(input_path)

    # --------------------------------------------------------
    # Basic dataset checks
    # --------------------------------------------------------

    required_columns = (
        ["user_id", "date", "role"]
        + ML_FEATURES
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{experiment_name}: "
            f"missing required columns: "
            f"{missing_columns}"
        )

    if len(df) != 300:
        raise ValueError(
            f"{experiment_name}: "
            f"expected 300 user-day rows, "
            f"found {len(df)}."
        )

    # --------------------------------------------------------
    # Critical ML feature NaN check
    # --------------------------------------------------------

    if df[ML_FEATURES].isna().any().any():

        missing_counts = (
            df[ML_FEATURES]
            .isna()
            .sum()
        )

        missing_counts = missing_counts[
            missing_counts > 0
        ]

        raise ValueError(
            f"{experiment_name}: "
            f"missing ML feature values found:\n"
            f"{missing_counts}"
        )

    prediction_parts = []

    # --------------------------------------------------------
    # Role-specific scoring
    # --------------------------------------------------------

    for role in sorted(df["role"].unique()):

        if role not in thresholds:
            raise ValueError(
                f"No calibrated threshold "
                f"found for role: {role}"
            )

        role_df = df[
            df["role"] == role
        ].copy()

        model_path = model_filename(role)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found for "
                f"{role}: {model_path}"
            )

        model_package = joblib.load(
            model_path
        )

        # ----------------------------------------------------
        # Validate model package
        # ----------------------------------------------------

        if "model" not in model_package:
            raise ValueError(
                f"Model package for {role} "
                f"does not contain 'model'."
            )

        if "features" not in model_package:
            raise ValueError(
                f"Model package for {role} "
                f"does not contain 'features'."
            )

        model_features = model_package[
            "features"
        ]

        if list(model_features) != list(ML_FEATURES):
            raise ValueError(
                f"Feature mismatch for {role}.\n"
                f"Model features: {model_features}\n"
                f"Current features: {ML_FEATURES}"
            )

        model = model_package["model"]

        X = role_df[ML_FEATURES]

        # ----------------------------------------------------
        # Isolation Forest scoring
        #
        # sklearn score_samples:
        # lower value = more abnormal
        #
        # Project definition:
        # anomaly_score = -score_samples
        #
        # Therefore:
        # higher anomaly_score = more unusual
        # ----------------------------------------------------

        raw_score = model.score_samples(X)

        role_df["anomaly_score"] = (
            -raw_score
        )

        threshold = thresholds[role]

        role_df["threshold"] = threshold

        # ----------------------------------------------------
        # Prediction
        #
        # 0 = Normal
        # 1 = Anomaly
        # ----------------------------------------------------

        role_df["prediction"] = (
            role_df["anomaly_score"]
            > threshold
        ).astype(int)

        prediction_parts.append(
            role_df
        )

    # --------------------------------------------------------
    # Combine all roles
    # --------------------------------------------------------

    predictions = pd.concat(
        prediction_parts,
        ignore_index=True,
    )

    predictions = predictions.sort_values(
        [
            "date",
            "role",
            "user_id",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(predictions) != len(df):
        raise ValueError(
            f"{experiment_name}: "
            "prediction row count changed "
            "during scoring."
        )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        PREDICTION_DIR
        / f"{experiment_name}_predictions.csv"
    )

    predictions.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # Console summary
    #
    # IMPORTANT:
    # This is prediction only.
    # Ground truth is NOT used here.
    # --------------------------------------------------------

    normal_count = int(
        (predictions["prediction"] == 0)
        .sum()
    )

    anomaly_count = int(
        (predictions["prediction"] == 1)
        .sum()
    )

    print(
        f"\n=== {experiment_name} ==="
    )

    print(
        f"User-days         : "
        f"{len(predictions)}"
    )

    print(
        f"Predicted normal  : "
        f"{normal_count}"
    )

    print(
        f"Predicted anomaly : "
        f"{anomaly_count}"
    )

    print(
        "\nPredicted anomalies per role:"
    )

    anomaly_by_role = (
        predictions[
            predictions["prediction"] == 1
        ]
        ["role"]
        .value_counts()
        .reindex(
            sorted(
                predictions["role"].unique()
            ),
            fill_value=0,
        )
    )

    print(anomaly_by_role)

    print(
        f"\nSaved: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    config = load_config()

    thresholds = load_thresholds()

    print(
        "=== Blind Experiment Scoring ==="
    )

    print(
        "\nLoaded frozen validation thresholds:"
    )

    for role in sorted(thresholds):
        print(
            f"{role:<20} "
            f"{thresholds[role]:.6f}"
        )

    # --------------------------------------------------------
    # Experiments are obtained from experiment_config.json
    #
    # experiment_1 = obvious
    # experiment_2 = moderate
    # experiment_3 = subtle
    # --------------------------------------------------------

    for experiment_name in (
        config["difficulty_levels"]
    ):

        score_experiment(
            experiment_name=experiment_name,
            thresholds=thresholds,
        )

    print(
        "\n=== Blind Scoring Complete ==="
    )

    print(
        "Ground truth was not used "
        "during prediction."
    )


if __name__ == "__main__":
    main()