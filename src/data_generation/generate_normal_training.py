import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROJECT_CONFIG_PATH = PROJECT_ROOT / "configs" / "project_config.json"
SYNTHETIC_CONFIG_PATH = PROJECT_ROOT / "configs" / "synthetic_data_config.json"

OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "normal_training.csv"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_working_days(start_date, number_of_days):
    dates = []
    current_date = pd.Timestamp(start_date)

    while len(dates) < number_of_days:
        if current_date.weekday() < 5:
            dates.append(current_date)
        current_date += pd.Timedelta(days=1)

    return dates


def hour_to_datetime(date, decimal_hour):
    decimal_hour = max(0.0, min(decimal_hour, 23.99))

    hour = int(decimal_hour)
    minute_float = (decimal_hour - hour) * 60
    minute = int(minute_float)
    second = int((minute_float - minute) * 60)

    return datetime(
        year=date.year,
        month=date.month,
        day=date.day,
        hour=hour,
        minute=minute,
        second=second,
    )


def create_employee_ids(role_config, count):
    prefix = role_config["user_prefix"]

    return [
        f"{prefix}{index:03d}"
        for index in range(1, count + 1)
    ]


def generate_known_ip(user_id):
    numeric_part = int(user_id[1:])

    role_prefix = user_id[0]

    subnet_map = {
        "T": 10,
        "C": 20,
        "M": 30,
    }

    subnet = subnet_map[role_prefix]

    return f"10.10.{subnet}.{numeric_part}"


def generate_user_day(
    rng,
    date,
    user_id,
    role,
    role_config,
):
    rows = []

    event_count = int(
        round(
            rng.normal(
                role_config["daily_event_count_mean"],
                role_config["daily_event_count_std"],
            )
        )
    )

    event_count = max(event_count, 5)

    start_hour = rng.normal(
        role_config["start_hour_mean"],
        role_config["start_hour_std"],
    )

    work_duration = rng.normal(
        role_config["work_duration_mean"],
        role_config["work_duration_std"],
    )

    end_hour = start_hour + work_duration

    login_time = hour_to_datetime(date, start_hour)
    logout_time = hour_to_datetime(date, end_hour)

    known_ip = generate_known_ip(user_id)

    # Login event
    rows.append(
        {
            "timestamp": login_time,
            "user_id": user_id,
            "role": role,
            "event_type": "login",
            "records_accessed": 0,
            "download_count": 0,
            "failed_login": 0,
            "account_type": "none",
            "ip_address": known_ip,
            "is_known_ip": 1,
        }
    )

    if rng.random() < role_config["failed_login_probability"]:
        failed_login_time = login_time + timedelta(
            seconds=int(rng.integers(1, 300))
        )

        rows.append(
            {
                "timestamp": failed_login_time,
                "user_id": user_id,
                "role": role,
                "event_type": "failed_login",
                "records_accessed": 0,
                "download_count": 0,
                "failed_login": 1,
                "account_type": "none",
                "ip_address": known_ip,
                "is_known_ip": 1,
            }
        )

    operational_event_count = max(event_count - 2, 1)

    event_types = list(
        role_config["event_probabilities"].keys()
    )

    event_probabilities = list(
        role_config["event_probabilities"].values()
    )

    work_seconds = max(
        int((logout_time - login_time).total_seconds()),
        1,
    )

    offsets = np.sort(
        rng.integers(
            low=1,
            high=work_seconds,
            size=operational_event_count,
        )
    )

    for offset in offsets:
        timestamp = login_time + timedelta(seconds=int(offset))

        event_type = rng.choice(
            event_types,
            p=event_probabilities,
        )

        records_accessed = 0
        download_count = 0
        account_type = "standard"

        if event_type == "customer_record_access":
            low, high = role_config["records_accessed_range"]

            records_accessed = int(
                rng.integers(low, high + 1)
            )

            if rng.random() < role_config["vip_probability"]:
                account_type = "vip"

        elif event_type == "download":
            low, high = role_config["download_count_range"]

            download_count = int(
                rng.integers(low, high + 1)
            )

        # failed_login = int(
        #     event_type == "customer_record_access"
        #     and rng.random()
        #     < role_config["failed_login_probability"]
        # )

        rows.append(
            {
                "timestamp": timestamp,
                "user_id": user_id,
                "role": role,
                "event_type": event_type,
                "records_accessed": records_accessed,
                "download_count": download_count,
                "failed_login": 0,
                "account_type": account_type,
                "ip_address": known_ip,
                "is_known_ip": 1,
            }
        )

    # Logout event
    rows.append(
        {
            "timestamp": logout_time,
            "user_id": user_id,
            "role": role,
            "event_type": "logout",
            "records_accessed": 0,
            "download_count": 0,
            "failed_login": 0,
            "account_type": "none",
            "ip_address": known_ip,
            "is_known_ip": 1,
        }
    )

    return rows


def main():
    project_config = load_json(PROJECT_CONFIG_PATH)
    synthetic_config = load_json(SYNTHETIC_CONFIG_PATH)

    seed = project_config["random_seed"]
    rng = np.random.default_rng(seed)

    training_config = synthetic_config["training"]

    working_days = get_working_days(
        training_config["start_date"],
        training_config["working_days"],
    )

    all_rows = []

    for role, role_config in synthetic_config["roles"].items():

        employee_ids = create_employee_ids(
            role_config,
            training_config["employees_per_role"],
        )

        for user_id in employee_ids:
            for date in working_days:
                rows = generate_user_day(
                    rng=rng,
                    date=date,
                    user_id=user_id,
                    role=role,
                    role_config=role_config,
                )

                all_rows.extend(rows)

    dataframe = pd.DataFrame(all_rows)

    dataframe = dataframe.sort_values(
        ["timestamp", "user_id"]
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("=== Normal Training Data Generated ===")
    print(f"Random seed      : {seed}")
    print(f"Employees        : {dataframe['user_id'].nunique()}")
    print(f"Roles            : {dataframe['role'].nunique()}")
    print(f"Working days     : {len(working_days)}")
    print(f"Raw audit events : {len(dataframe):,}")
    print(f"Output           : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()