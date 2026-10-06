from pathlib import Path

import pandas as pd

from src.features.build_user_day_features import build_user_day_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "experiment_1_test.csv"
)

GT_PATH = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "experiment_1_ground_truth.csv"
)


def main():
    raw = pd.read_csv(RAW_PATH)
    gt = pd.read_csv(GT_PATH)

    features = build_user_day_features(raw)
    
    gt["date"] = pd.to_datetime(gt["date"]).dt.strftime("%Y-%m-%d")
    features["date"] = pd.to_datetime(features["date"]).dt.strftime("%Y-%m-%d")

    gt["user_id"] = gt["user_id"].astype(str)
    features["user_id"] = features["user_id"].astype(str)

    gt["role"] = gt["role"].astype(str).str.strip()
    features["role"] = features["role"].astype(str).str.strip()
    

    merged = gt.merge(
        features,
        on=["user_id", "date", "role"],
        how="left",
    )

    columns = [
        "case_id",
        "role",
        "scenario",
        "target_download_total",
        "download_total",
        "target_records_accessed_total",
        "records_accessed_total",
        "target_last_activity_hour",
        "last_activity_hour",
    ]

    print(
        merged[columns]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()