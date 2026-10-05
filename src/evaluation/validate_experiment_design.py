import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiment_config.json"
)

REFERENCE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "normal_feature_reference.csv"
)


ROLES = [
    "Teller",
    "Customer Service",
    "Manager",
]

SCENARIOS = [
    "unusual_export",
    "unusual_record_access",
    "temporal_combined",
]

DIFFICULTIES = [
    "obvious",
    "moderate",
    "subtle",
]


def get_reference(
    reference,
    role,
    feature,
):
    row = reference[
        (reference["role"] == role)
        & (reference["feature"] == feature)
    ]

    if len(row) != 1:
        raise ValueError(
            f"Reference row not unique: "
            f"{role} / {feature}"
        )

    return row.iloc[0]


def print_range(
    role,
    scenario,
    difficulty,
    feature,
    target_range,
    reference,
):
    ref = get_reference(
        reference,
        role,
        feature,
    )

    low, high = target_range

    print(
        f"{scenario:24} "
        f"{difficulty:10} "
        f"{feature:24} "
        f"target={low:7.2f}-{high:7.2f} | "
        f"p95={ref['p95']:7.2f} "
        f"p99={ref['p99']:7.2f} "
        f"max={ref['max']:7.2f}"
    )


def main():
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    reference = pd.read_csv(
        REFERENCE_PATH
    )

    design = config[
        "experiment_design"
    ]

    allocation = design[
        "scenario_allocation_per_role"
    ]

    allocated_total = sum(
        allocation.values()
    )

    expected_total = design[
        "anomaly_user_days_per_role"
    ]

    if allocated_total != expected_total:
        raise ValueError(
            "Scenario allocation does not equal "
            "anomaly_user_days_per_role."
        )

    print(
        "=== Experiment Design Validation ==="
    )

    print(
        f"Anomaly user-days per role: "
        f"{expected_total}"
    )

    print(
        f"Total anomaly user-days per experiment: "
        f"{expected_total * len(ROLES)}"
    )

    print()

    for role in ROLES:

        print("=" * 130)
        print(f"ROLE: {role}")
        print("=" * 130)

        role_config = config[
            "scenario_targets"
        ][role]

        for difficulty in DIFFICULTIES:

            export_range = (
                role_config[
                    "unusual_export"
                ][difficulty]
            )

            print_range(
                role,
                "unusual_export",
                difficulty,
                "download_total",
                export_range,
                reference,
            )

            access_range = (
                role_config[
                    "unusual_record_access"
                ][difficulty]
            )

            print_range(
                role,
                "unusual_record_access",
                difficulty,
                "records_accessed_total",
                access_range,
                reference,
            )

            combined = (
                role_config[
                    "temporal_combined"
                ][difficulty]
            )

            for feature in [
                "last_activity_hour",
                "download_total",
                "records_accessed_total",
            ]:
                print_range(
                    role,
                    "temporal_combined",
                    difficulty,
                    feature,
                    combined[feature],
                    reference,
                )

        print()

    print(
        "Experiment configuration "
        "validated successfully."
    )


if __name__ == "__main__":
    main()