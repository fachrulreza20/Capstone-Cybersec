import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_config.json"
RAW_BASE_PATH = PROJECT_ROOT / "data" / "raw" / "normal_test.csv"
CASES_PATH = PROJECT_ROOT / "data" / "ground_truth" / "experiment_cases.csv"

RAW_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
GROUND_TRUTH_DIR = PROJECT_ROOT / "data" / "ground_truth"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def random_integer_target(rng, target_range):
    low, high = target_range
    return int(rng.integers(low, high + 1))


def random_float_target(rng, target_range):
    low, high = target_range
    return float(rng.uniform(low, high))


def decimal_hour_to_timestamp(date_value, decimal_hour):
    date = pd.Timestamp(date_value)

    hour = int(decimal_hour)
    minute_float = (decimal_hour - hour) * 60
    minute = int(minute_float)
    second = int((minute_float - minute) * 60)

    return pd.Timestamp(
        year=date.year,
        month=date.month,
        day=date.day,
        hour=hour,
        minute=minute,
        second=second,
    )


def add_download_events(
    df,
    user_id,
    date_value,
    target_total,
):
    date_value = pd.Timestamp(date_value).date()

    mask = (
        (df["user_id"] == user_id)
        & (df["timestamp"].dt.date == date_value)
    )

    current_total = int(
        df.loc[mask, "download_count"].sum()
    )

    amount_needed = target_total - current_total

    if amount_needed <= 0:
        return df

    user_day = df.loc[mask].copy()

    last_time = user_day["timestamp"].max()

    template = user_day.iloc[-1].copy()

    new_row = template.copy()

    # Insert before/around the end of the existing day.
    new_row["timestamp"] = last_time - timedelta(seconds=30)
    new_row["event_type"] = "download"
    new_row["records_accessed"] = 0
    new_row["download_count"] = amount_needed
    new_row["failed_login"] = 0
    new_row["account_type"] = "standard"

    return pd.concat(
        [df, pd.DataFrame([new_row])],
        ignore_index=True,
    )


def add_record_access_events(
    df,
    user_id,
    date_value,
    target_total,
):
    date_value = pd.Timestamp(date_value).date()

    mask = (
        (df["user_id"] == user_id)
        & (df["timestamp"].dt.date == date_value)
    )

    current_total = int(
        df.loc[mask, "records_accessed"].sum()
    )

    amount_needed = target_total - current_total

    if amount_needed <= 0:
        return df

    user_day = df.loc[mask].copy()

    last_time = user_day["timestamp"].max()

    template = user_day.iloc[-1].copy()

    new_row = template.copy()

    new_row["timestamp"] = last_time - timedelta(seconds=20)
    new_row["event_type"] = "customer_record_access"
    new_row["records_accessed"] = amount_needed
    new_row["download_count"] = 0
    new_row["failed_login"] = 0
    new_row["account_type"] = "standard"

    return pd.concat(
        [df, pd.DataFrame([new_row])],
        ignore_index=True,
    )


def extend_last_activity(
    df,
    user_id,
    date_value,
    target_hour,
):
    date_value = pd.Timestamp(date_value).date()

    mask = (
        (df["user_id"] == user_id)
        & (df["timestamp"].dt.date == date_value)
    )

    user_day = df.loc[mask].copy()

    target_timestamp = decimal_hour_to_timestamp(
        date_value,
        target_hour,
    )

    current_last = user_day["timestamp"].max()

    if target_timestamp <= current_last:
        return df

    template = user_day.iloc[-1].copy()

    new_row = template.copy()

    new_row["timestamp"] = target_timestamp
    new_row["event_type"] = "logout"
    new_row["records_accessed"] = 0
    new_row["download_count"] = 0
    new_row["failed_login"] = 0
    new_row["account_type"] = "none"

    return pd.concat(
        [df, pd.DataFrame([new_row])],
        ignore_index=True,
    )


def generate_experiment(
    base_df,
    cases,
    config,
    experiment_name,
    difficulty,
    rng,
):
    df = base_df.copy()

    ground_truth_rows = []

    for _, case in cases.iterrows():
        role = case["role"]
        scenario = case["scenario"]
        user_id = case["user_id"]
        date_value = case["date"]

        target_config = (
            config["scenario_targets"]
            [role][scenario][difficulty]
        )

        target_download = None
        target_records = None
        target_last_hour = None

        if scenario == "unusual_export":
            target_download = random_integer_target(
                rng,
                target_config,
            )

            df = add_download_events(
                df,
                user_id,
                date_value,
                target_download,
            )

        elif scenario == "unusual_record_access":
            target_records = random_integer_target(
                rng,
                target_config,
            )

            df = add_record_access_events(
                df,
                user_id,
                date_value,
                target_records,
            )

        elif scenario == "temporal_combined":
            target_download = random_integer_target(
                rng,
                target_config["download_total"],
            )

            target_records = random_integer_target(
                rng,
                target_config["records_accessed_total"],
            )

            target_last_hour = random_float_target(
                rng,
                target_config["last_activity_hour"],
            )

            df = add_download_events(
                df,
                user_id,
                date_value,
                target_download,
            )

            df = add_record_access_events(
                df,
                user_id,
                date_value,
                target_records,
            )

            df = extend_last_activity(
                df,
                user_id,
                date_value,
                target_last_hour,
            )

        ground_truth_rows.append(
            {
                "case_id": case["case_id"],
                "user_id": user_id,
                "date": date_value,
                "role": role,
                "scenario": scenario,
                "difficulty": difficulty,
                "ground_truth": 1,
                "target_download_total": target_download,
                "target_records_accessed_total": target_records,
                "target_last_activity_hour": target_last_hour,
            }
        )

    df = df.sort_values(
        ["timestamp", "user_id"]
    ).reset_index(drop=True)

    ground_truth = pd.DataFrame(
        ground_truth_rows
    )

    raw_output = (
        RAW_OUTPUT_DIR
        / f"{experiment_name}_test.csv"
    )

    gt_output = (
        GROUND_TRUTH_DIR
        / f"{experiment_name}_ground_truth.csv"
    )

    df.to_csv(raw_output, index=False)
    ground_truth.to_csv(gt_output, index=False)

    print(
        f"{experiment_name}: "
        f"{difficulty} | "
        f"raw events={len(df):,} | "
        f"anomaly cases={len(ground_truth)}"
    )


def main():
    config = load_config()

    base_df = pd.read_csv(
        RAW_BASE_PATH,
        parse_dates=["timestamp"],
    )

    cases = pd.read_csv(CASES_PATH)

    seed = (
        config["experiment_design"]
        ["injection_seed"]
    )

    RAW_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    GROUND_TRUTH_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=== Generating Experiments ===")
    print(f"Injection seed: {seed}\n")

    for index, (
        experiment_name,
        difficulty,
    ) in enumerate(
        config["difficulty_levels"].items()
    ):
        # Independent deterministic RNG per experiment.
        rng = np.random.default_rng(
            seed + index
        )

        generate_experiment(
            base_df=base_df,
            cases=cases,
            config=config,
            experiment_name=experiment_name,
            difficulty=difficulty,
            rng=rng,
        )

    print("\nAll experiment datasets generated.")


if __name__ == "__main__":
    main()