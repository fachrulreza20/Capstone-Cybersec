from pathlib import Path

import numpy as np
import pandas as pd

from src.data_generation.generate_normal_training import (
    create_employee_ids,
    generate_user_day,
    get_working_days,
    load_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROJECT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "project_config.json"
)

SYNTHETIC_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "synthetic_data_config.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "normal_validation.csv"
)


def main():
    project_config = load_json(
        PROJECT_CONFIG_PATH
    )

    synthetic_config = load_json(
        SYNTHETIC_CONFIG_PATH
    )

    seed = project_config["validation_seed"]

    rng = np.random.default_rng(seed)

    validation_config = (
        synthetic_config["validation"]
    )

    working_days = get_working_days(
        validation_config["start_date"],
        validation_config["working_days"],
    )

    all_rows = []

    for role, role_config in (
        synthetic_config["roles"].items()
    ):

        employee_ids = create_employee_ids(
            role_config,
            validation_config[
                "employees_per_role"
            ],
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

    df = pd.DataFrame(all_rows)

    df = df.sort_values(
        ["timestamp", "user_id"]
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "=== Normal Validation Data Generated ==="
    )

    print(f"Random seed      : {seed}")
    print(
        f"Employees        : "
        f"{df['user_id'].nunique()}"
    )
    print(
        f"Working days     : "
        f"{len(working_days)}"
    )
    print(
        f"Raw audit events : "
        f"{len(df):,}"
    )
    print(f"Output           : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()