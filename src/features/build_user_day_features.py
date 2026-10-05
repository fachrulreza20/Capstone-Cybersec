from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "normal_training.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "normal_training_user_day.csv"
)


def decimal_hour(timestamp):
    return (
        timestamp.hour
        + timestamp.minute / 60
        + timestamp.second / 3600
    )


def build_user_day_features(df):
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["date"] = df["timestamp"].dt.date

    df["is_transaction"] = (
        df["event_type"] == "transaction"
    ).astype(int)

    df["is_customer_record_access"] = (
        df["event_type"] == "customer_record_access"
    ).astype(int)

    df["is_download"] = (
        df["event_type"] == "download"
    ).astype(int)

    df["is_vip_access"] = (
        (df["event_type"] == "customer_record_access")
        & (df["account_type"] == "vip")
    ).astype(int)

    grouped = (
        df.groupby(
            ["user_id", "date", "role"],
            as_index=False,
        )
        .agg(
            event_count=("event_type", "size"),
            failed_login_count=("failed_login", "sum"),
            transaction_count=("is_transaction", "sum"),
            customer_record_access_count=(
                "is_customer_record_access",
                "sum",
            ),
            download_event_count=("is_download", "sum"),
            download_total=("download_count", "sum"),
            records_accessed_total=(
                "records_accessed",
                "sum",
            ),
            vip_access_count=("is_vip_access", "sum"),
            unknown_ip_count=(
                "is_known_ip",
                lambda values: (values == 0).sum(),
            ),
            unique_ip_count=("ip_address", "nunique"),
            first_activity=("timestamp", "min"),
            last_activity=("timestamp", "max"),
        )
    )

    grouped["first_activity_hour"] = (
        grouped["first_activity"].apply(decimal_hour)
    )

    grouped["last_activity_hour"] = (
        grouped["last_activity"].apply(decimal_hour)
    )

    grouped["activity_duration_hours"] = (
        (
            grouped["last_activity"]
            - grouped["first_activity"]
        ).dt.total_seconds()
        / 3600
    )

    grouped = grouped.drop(
        columns=[
            "first_activity",
            "last_activity",
        ]
    )

    return grouped


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

    print("=== User-Day Features Generated ===")
    print(f"Raw events        : {len(df):,}")
    print(f"User-day rows     : {len(features):,}")
    print(
        f"Unique employees  : "
        f"{features['user_id'].nunique()}"
    )
    print(
        f"Unique dates      : "
        f"{features['date'].nunique()}"
    )
    print(f"Output            : {OUTPUT_PATH}")

    print("\n=== Feature Columns ===")
    for column in features.columns:
        print(column)


if __name__ == "__main__":
    main()