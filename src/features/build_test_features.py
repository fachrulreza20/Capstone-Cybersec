from pathlib import Path

import pandas as pd

from src.features.build_user_day_features import (
    build_user_day_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "normal_test.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "normal_test_user_day.csv"
)


def main():
    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["timestamp"],
    )

    features = build_user_day_features(df)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("=== Normal Test User-Day Features ===")

    print(
        f"Raw events     : {len(df):,}"
    )

    print(
        f"User-day rows  : {len(features):,}"
    )

    print(
        f"Unique employees: "
        f"{features['user_id'].nunique()}"
    )

    print(
        f"Unique dates   : "
        f"{features['date'].nunique()}"
    )

    print("\nRows per role:")

    print(
        features["role"]
        .value_counts()
        .sort_index()
    )

    print(
        f"\nOutput: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()