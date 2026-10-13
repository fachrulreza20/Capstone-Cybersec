import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.robustness.replication_runner import (
    generate_normal_test,
    load_json,
)
from src.pipeline.detection_pipeline import (
    ROLE_CONTEXT,
    run_raw_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SYNTHETIC_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "synthetic_data_config.json"
)

EXPERIMENT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "experiment_config.json"
)

LLM_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "llm_config.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "robustness"
)

SAMPLE_CSV_PATH = (
    OUTPUT_DIR / "llm_control_sample.csv"
)

CONTEXT_PATH = (
    OUTPUT_DIR / "llm_control_context.json"
)


ROLES = [
    "Teller",
    "Customer Service",
    "Manager",
]

SAMPLE_SEED = 51001

CONTROL_DATA_SEEDS = range(
    10001,
    10051,
)

SAMPLES_PER_ROLE_PER_GROUP = 10

FORBIDDEN_LLM_FIELDS = {
    "true_label",
    "case_id",
    "scenario",
    "difficulty",
    "experiment_difficulty",
    "control_group",
    "replication",
    "data_seed",
}


def python_value(value):
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def validate_no_forbidden_fields(data):
    if isinstance(data, dict):
        for key, value in data.items():

            if key in FORBIDDEN_LLM_FIELDS:
                raise ValueError(
                    "Forbidden diagnostic field leaked "
                    f"into LLM context: {key}"
                )

            validate_no_forbidden_fields(value)

    elif isinstance(data, list):
        for item in data:
            validate_no_forbidden_fields(item)


def collect_control_pool(
    synthetic_config,
    experiment_config,
):
    """
    Deterministically regenerate all 50 normal-only
    robustness datasets.

    Every user-day has ground truth normal because no
    anomaly injection occurs in these datasets.
    """

    pool_parts = []

    for replication, data_seed in enumerate(
        CONTROL_DATA_SEEDS,
        start=1,
    ):
        print(
            f"[CONTROL] Regenerating "
            f"{replication:02d}/50 "
            f"(seed={data_seed})"
        )

        raw_df = generate_normal_test(
            seed=data_seed,
            synthetic_config=synthetic_config,
            experiment_config=experiment_config,
        )

        result = run_raw_pipeline(
            raw_df
        )

        combined = result["combined"].copy()

        if len(combined) != 300:
            raise ValueError(
                f"Control replication {replication}: "
                f"expected 300 user-days, "
                f"found {len(combined)}."
            )

        combined["replication"] = replication
        combined["data_seed"] = data_seed

        # Diagnostic metadata only.
        # This will NEVER be passed to the LLM.
        combined["true_label"] = 0

        combined["control_group"] = np.where(
            combined["candidate"].astype(int) == 1,
            "flagged_normal",
            "unflagged_normal",
        )

        pool_parts.append(
            combined
        )

    pool = pd.concat(
        pool_parts,
        ignore_index=True,
    )

    if len(pool) != 15000:
        raise ValueError(
            f"Expected 15,000 normal control "
            f"user-days, found {len(pool)}."
        )

    if pool["true_label"].sum() != 0:
        raise ValueError(
            "Control pool unexpectedly contains "
            "positive ground-truth labels."
        )

    return pool


def print_pool_summary(pool):
    print("\n=== CONTROL POOL ===")
    print(
        f"Total normal user-days : {len(pool)}"
    )

    summary = (
        pool.groupby(
            ["control_group", "role"]
        )
        .size()
        .unstack(fill_value=0)
    )

    print(summary.to_string())

    for group in [
        "flagged_normal",
        "unflagged_normal",
    ]:
        for role in ROLES:
            count = len(
                pool[
                    (pool["control_group"] == group)
                    & (pool["role"] == role)
                ]
            )

            if count < SAMPLES_PER_ROLE_PER_GROUP:
                raise ValueError(
                    f"Insufficient pool for "
                    f"{group}/{role}: {count}"
                )


def select_balanced_sample(pool):
    """
    Fixed-seed role-balanced sample:

    flagged normal:
        10 Teller
        10 Customer Service
        10 Manager

    unflagged normal:
        10 Teller
        10 Customer Service
        10 Manager

    Total = 60.
    """

    rng = np.random.default_rng(
        SAMPLE_SEED
    )

    selected_parts = []

    groups = [
        "flagged_normal",
        "unflagged_normal",
    ]

    for group in groups:
        for role in ROLES:

            eligible = pool[
                (pool["control_group"] == group)
                & (pool["role"] == role)
            ].copy()

            chosen_positions = rng.choice(
                len(eligible),
                size=SAMPLES_PER_ROLE_PER_GROUP,
                replace=False,
            )

            chosen = eligible.iloc[
                chosen_positions
            ].copy()

            selected_parts.append(
                chosen
            )

    sample = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    sample = sample.sort_values(
        [
            "control_group",
            "role",
            "replication",
            "user_id",
            "date",
        ]
    ).reset_index(drop=True)

    sample.insert(
        0,
        "control_case_id",
        [
            f"CTRL{index:03d}"
            for index in range(
                1,
                len(sample) + 1,
            )
        ],
    )

    if len(sample) != 60:
        raise ValueError(
            f"Expected 60 control cases, "
            f"found {len(sample)}."
        )

    return sample


def build_llm_package(row, feature_names):
    """
    Build exactly the same substantive LLM evidence
    structure used by the existing project.

    Diagnostic metadata is deliberately excluded.
    """

    behaviour = {}

    for feature in feature_names:

        if feature not in row.index:
            raise ValueError(
                f"Configured behaviour feature "
                f"not found: {feature}"
            )

        behaviour[feature] = python_value(
            row[feature]
        )

    package = {
        "employee_context": {
            "user_id": row["user_id"],
            "date": row["date"],
            "role": row["role"],
            "role_context": ROLE_CONTEXT.get(
                row["role"],
                "No role context available.",
            ),
        },

        "behaviour_summary": behaviour,

        "ml_evidence": {
            "anomaly_score": python_value(
                row["anomaly_score"]
            ),
            "threshold": python_value(
                row["threshold"]
            ),
            "prediction": (
                "Anomaly"
                if int(row["prediction"]) == 1
                else "Normal"
            ),
        },

        "rule_evidence": {
            "after_hours": bool(
                int(row["rule_after_hours"])
            ),
            "large_export": bool(
                int(row["rule_large_export"])
            ),
            "high_record_access": bool(
                int(row["rule_high_record_access"])
            ),
            "rules_triggered": int(
                row["rules_triggered"]
            ),
            "prediction": (
                "Anomaly"
                if int(row["rule_prediction"]) == 1
                else "Normal"
            ),
        },
    }

    validate_no_forbidden_fields(
        package
    )

    return package


def save_sample_csv(sample):
    """
    Internal audit file.

    This file MAY contain diagnostic metadata because
    it is never sent to the LLM.
    """

    columns = [
        "control_case_id",
        "control_group",
        "replication",
        "data_seed",
        "user_id",
        "date",
        "role",
        "true_label",
        "prediction",
        "rule_prediction",
        "candidate",
        "anomaly_score",
        "threshold",
        "rules_triggered",
        "rule_after_hours",
        "rule_large_export",
        "rule_high_record_access",
    ]

    SAMPLE_CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample[
        columns
    ].to_csv(
        SAMPLE_CSV_PATH,
        index=False,
    )


def save_context(sample, llm_config):
    feature_names = llm_config[
        "include_behaviour_features"
    ]

    cases = []

    for _, row in sample.iterrows():

        package = build_llm_package(
            row,
            feature_names,
        )

        cases.append(
            {
                # control_case_id stays OUTSIDE
                # candidate_input.
                "control_case_id": (
                    row["control_case_id"]
                ),
                "candidate_input": package,
            }
        )

    # Validate candidate inputs separately.
    for case in cases:
        validate_no_forbidden_fields(
            case["candidate_input"]
        )

    output = {
        "study": "llm_normal_control_validation",
        "sample_seed": SAMPLE_SEED,
        "sample_size": len(cases),
        "sampling_design": (
            "30 flagged-normal and 30 "
            "unflagged-normal synthetic user-days; "
            "10 per banking role in each group."
        ),
        "important_note": (
            "control_group and true_label are retained "
            "only in the separate audit CSV and are not "
            "included in candidate_input sent to the LLM."
        ),
        "cases": cases,
    }

    with open(
        CONTEXT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )


def validate_sample(sample):
    print("\n=== SAMPLE VALIDATION ===")

    if len(sample) != 60:
        raise ValueError(
            "Sample size is not 60."
        )

    if not (
        sample["true_label"] == 0
    ).all():
        raise ValueError(
            "Non-normal ground truth found."
        )

    for group in [
        "flagged_normal",
        "unflagged_normal",
    ]:
        for role in ROLES:

            subset = sample[
                (sample["control_group"] == group)
                & (sample["role"] == role)
            ]

            if len(subset) != 10:
                raise ValueError(
                    f"{group}/{role}: "
                    f"expected 10, found {len(subset)}."
                )

    flagged = sample[
        sample["control_group"]
        == "flagged_normal"
    ]

    unflagged = sample[
        sample["control_group"]
        == "unflagged_normal"
    ]

    if not (
        flagged["candidate"].astype(int) == 1
    ).all():
        raise ValueError(
            "Flagged-normal group contains "
            "non-candidate rows."
        )

    if not (
        unflagged["candidate"].astype(int) == 0
    ).all():
        raise ValueError(
            "Unflagged-normal group contains "
            "candidate rows."
        )

    if not (
        (
            flagged["prediction"].astype(int) == 1
        )
        |
        (
            flagged[
                "rule_prediction"
            ].astype(int) == 1
        )
    ).all():
        raise ValueError(
            "Invalid flagged-normal selection."
        )

    if not (
        (
            unflagged["prediction"].astype(int) == 0
        )
        &
        (
            unflagged[
                "rule_prediction"
            ].astype(int) == 0
        )
    ).all():
        raise ValueError(
            "Invalid unflagged-normal selection."
        )

    print("Total cases             : 60 PASS")
    print("Ground truth normal     : 60/60 PASS")
    print("Flagged normal          : 30/30 PASS")
    print("Unflagged normal        : 30/30 PASS")
    print("Role balance            : PASS")
    print("Ground-truth leakage    : NONE")
    print("Difficulty leakage      : NONE")
    print("Scenario leakage        : NONE")
    print("LLM API calls           : 0")


def main():
    print(
        "=== Stage 30D.3A "
        "Prepare LLM Normal-Control Sample ==="
    )

    synthetic_config = load_json(
        SYNTHETIC_CONFIG_PATH
    )

    experiment_config = load_json(
        EXPERIMENT_CONFIG_PATH
    )

    llm_config = load_json(
        LLM_CONFIG_PATH
    )

    if (
        llm_config.get(
            "include_raw_timeline"
        )
        is not False
    ):
        raise ValueError(
            "LLM raw timeline must remain excluded."
        )

    pool = collect_control_pool(
        synthetic_config,
        experiment_config,
    )

    print_pool_summary(
        pool
    )

    sample = select_balanced_sample(
        pool
    )

    validate_sample(
        sample
    )

    save_sample_csv(
        sample
    )

    save_context(
        sample,
        llm_config,
    )

    print("\n=== OUTPUTS ===")
    print(
        f"Audit sample : {SAMPLE_CSV_PATH}"
    )
    print(
        f"LLM context  : {CONTEXT_PATH}"
    )

    print(
        "\nSTAGE 30D.3A: PASS"
    )


if __name__ == "__main__":
    main()