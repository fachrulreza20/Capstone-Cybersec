from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "normal_training_user_day.csv"
)


IDENTIFIER_COLUMNS = [
    "user_id",
    "date",
    "role",
]


def main():
    df = pd.read_csv(DATA_PATH)

    feature_columns = [
        column
        for column in df.columns
        if column not in IDENTIFIER_COLUMNS
    ]

    print("=== User-Day Dataset ===")
    print(f"Rows             : {len(df)}")
    print(f"Employees        : {df['user_id'].nunique()}")
    print(f"Dates            : {df['date'].nunique()}")
    print()

    print("=== Rows per Role ===")
    print(df["role"].value_counts())
    print()

    print("=== Missing Values ===")
    print(df.isna().sum())
    print()

    print("=== Duplicate User-Day Checks ===")

    duplicate_count = df.duplicated(
        subset=["user_id", "date"]
    ).sum()

    print(f"Duplicate user-days: {duplicate_count}")
    print()

    print("=== Feature Summary ===")

    print(
        df[feature_columns]
        .describe()
        .T
        .round(3)
    )

    print("\n=== Unique Values per Feature ===")

    for column in feature_columns:
        print(
            f"{column:30s} "
            f"{df[column].nunique()}"
        )

    print("\n=== Mean Features by Role ===")

    print(
        df.groupby("role")[feature_columns]
        .mean()
        .round(3)
        .T
    )

    print("\n=== Standard Deviation by Role ===")

    print(
        df.groupby("role")[feature_columns]
        .std()
        .round(3)
        .T
    )

    print("\n=== First 10 User-Day Rows ===")

    print(
        df.head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()