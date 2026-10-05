from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "normal_training.csv"
)


def main():
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"],
    )

    print("=== Dataset Shape ===")
    print(df.shape)

    print("\n=== Columns ===")
    print(df.columns.tolist())

    print("\n=== First 10 Rows ===")
    print(df.head(10).to_string(index=False))

    print("\n=== Missing Values ===")
    print(df.isna().sum())

    print("\n=== Unique Employees ===")
    print(df["user_id"].nunique())

    print("\n=== Role Distribution (Raw Events) ===")
    print(df["role"].value_counts())

    print("\n=== Event Type Distribution ===")
    print(df["event_type"].value_counts())

    print("\n=== Account Type Distribution ===")
    print(df["account_type"].value_counts())

    print("\n=== Known IP Distribution ===")
    print(df["is_known_ip"].value_counts())

    print("\n=== Failed Login Distribution ===")
    print(df["failed_login"].value_counts())

    print("\n=== Date Range ===")
    print("Start:", df["timestamp"].min())
    print("End  :", df["timestamp"].max())

    print("\n=== Numerical Summary ===")
    print(
        df[
            [
                "records_accessed",
                "download_count",
                "failed_login",
                "is_known_ip",
            ]
        ].describe()
    )


    print("\n=== Mean Values by Role ===")

    role_summary = (
        df.groupby("role")[
            [
                "records_accessed",
                "download_count",
                "failed_login",
            ]
        ]
        .mean()
        .round(3)
    )

    print(role_summary)

    print("\n=== Event Type Counts by Role ===")

    role_event_counts = pd.crosstab(
        df["role"],
        df["event_type"],
    )

    print(role_event_counts)


if __name__ == "__main__":
    main()