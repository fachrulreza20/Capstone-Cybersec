import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiment_config.json"
)

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "normal_test_user_day.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "experiment_cases.csv"
)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def is_eligible(row, scenario, subtle_target):
    if scenario == "unusual_export":
        return (
            row["download_total"]
            < subtle_target[0]
        )

    if scenario == "unusual_record_access":
        return (
            row["records_accessed_total"]
            < subtle_target[0]
        )

    if scenario == "temporal_combined":
        return (
            row["last_activity_hour"]
            < subtle_target["last_activity_hour"][0]
            and row["download_total"]
            < subtle_target["download_total"][0]
            and row["records_accessed_total"]
            < subtle_target["records_accessed_total"][0]
        )

    raise ValueError(
        f"Unknown scenario: {scenario}"
    )


def main():
    config = load_config()

    design = config["experiment_design"]
    targets = config["scenario_targets"]

    seed = design["case_selection_seed"]
    allocation = design[
        "scenario_allocation_per_role"
    ]

    rng = np.random.default_rng(seed)

    df = pd.read_csv(INPUT_PATH)

    selected_rows = []

    for role in sorted(df["role"].unique()):
        role_df = df[
            df["role"] == role
        ].copy()

        # Prevent the same user-day from being selected
        # for more than one scenario.
        available_df = role_df.copy()

        for scenario, count in allocation.items():
            subtle_target = (
                targets[role][scenario]["subtle"]
            )

            eligible_mask = available_df.apply(
                lambda row: is_eligible(
                    row,
                    scenario,
                    subtle_target,
                ),
                axis=1,
            )

            eligible_df = available_df[
                eligible_mask
            ].copy()

            if len(eligible_df) < count:
                raise ValueError(
                    f"Not enough eligible cases for "
                    f"{role} / {scenario}. "
                    f"Needed {count}, "
                    f"available {len(eligible_df)}."
                )

            chosen_indices = rng.choice(
                eligible_df.index.to_numpy(),
                size=count,
                replace=False,
            )

            chosen = available_df.loc[
                chosen_indices
            ].copy()

            chosen["scenario"] = scenario

            selected_rows.append(
                chosen[
                    [
                        "user_id",
                        "date",
                        "role",
                        "scenario",
                        "download_total",
                        "records_accessed_total",
                        "last_activity_hour",
                    ]
                ]
            )

            # Remove selected user-days so they cannot
            # appear in another scenario.
            available_df = available_df.drop(
                index=chosen_indices
            )

    cases = pd.concat(
        selected_rows,
        ignore_index=True,
    )

    cases = cases.sort_values(
        ["role", "date", "user_id"]
    ).reset_index(drop=True)

    cases.insert(
        0,
        "case_id",
        [
            f"CASE{number:03d}"
            for number in range(
                1,
                len(cases) + 1,
            )
        ],
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("=== Eligible Experiment Cases Selected ===")
    print(f"Selection seed : {seed}")
    print(f"Total cases    : {len(cases)}")

    print("\nCases per role:")
    print(
        cases["role"]
        .value_counts()
        .sort_index()
    )

    print("\nScenario allocation:")
    print(
        cases.groupby(
            ["role", "scenario"]
        ).size()
    )

    print(f"\nOutput: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()