import json
from pathlib import Path

import numpy as np

from src.data_generation.generate_normal_training import (
    generate_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SYNTHETIC_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "synthetic_data_config.json"
)

EXPERIMENT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiment_config.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "normal_test.csv"
)


def main():
    # Load normal synthetic banking behaviour configuration.
    with open(
        SYNTHETIC_CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        synthetic_config = json.load(file)

    # Load test-period configuration.
    with open(
        EXPERIMENT_CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        experiment_config = json.load(file)

    test_config = experiment_config["test_base"]

    random_seed = test_config["random_seed"]
    start_date = test_config["start_date"]
    working_days = test_config["working_days"]

    # IMPORTANT:
    # Copy the normal training configuration so that the
    # underlying role behaviour remains identical.
    test_synthetic_config = synthetic_config.copy()

    test_synthetic_config["training"] = (
        synthetic_config["training"].copy()
    )

    test_synthetic_config["training"]["start_date"] = (
        start_date
    )

    test_synthetic_config["training"]["working_days"] = (
        working_days
    )

    rng = np.random.default_rng(random_seed)

    # Generate NORMAL test events using exactly the same
    # behavioural assumptions as training/validation.
    df = generate_dataset(
        config=test_synthetic_config,
        rng=rng,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("=== Normal Test Data Generated ===")
    print(f"Random seed      : {random_seed}")
    print(
        f"Employees        : "
        f"{df['user_id'].nunique()}"
    )
    print(
        f"Roles            : "
        f"{df['role'].nunique()}"
    )
    print(
        f"Working days     : "
        f"{working_days}"
    )
    print(
        f"Raw audit events : "
        f"{len(df):,}"
    )
    print(f"Output           : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()