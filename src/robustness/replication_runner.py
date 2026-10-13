import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_generation.generate_normal_training import (
    create_employee_ids,
    generate_user_day,
    get_working_days,
)
from src.data_generation.generate_experiments import (
    add_download_events,
    add_record_access_events,
    extend_last_activity,
    random_float_target,
    random_integer_target,
)
from src.data_generation.select_experiment_cases import (
    is_eligible,
)
from src.pipeline.detection_pipeline import (
    run_raw_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SYNTHETIC_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "synthetic_data_config.json"
)

EXPERIMENT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "experiment_config.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "robustness"
)

METRICS_PATH = (
    OUTPUT_DIR / "replication_metrics.csv"
)

CASE_AUDIT_PATH = (
    OUTPUT_DIR / "replication_case_audit.csv"
)


CONDITIONS = {
    "control": {
        "difficulty": None,
        "seed_base": 10000,
    },
    "obvious": {
        "difficulty": "obvious",
        "seed_base": 20000,
    },
    "moderate": {
        "difficulty": "moderate",
        "seed_base": 30000,
    },
    "subtle": {
        "difficulty": "subtle",
        "seed_base": 40000,
    },
}


KEY_COLUMNS = [
    "user_id",
    "date",
    "role",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def generate_normal_test(
    seed,
    synthetic_config,
    experiment_config,
):
    """
    Generate one independent normal test dataset in memory.

    This intentionally reuses the same generate_user_day()
    implementation as the original project, but does not
    overwrite data/raw/normal_test.csv.
    """

    rng = np.random.default_rng(seed)

    test_config = experiment_config["test_base"]

    working_days = get_working_days(
        test_config["start_date"],
        test_config["working_days"],
    )

    employees_per_role = (
        synthetic_config["training"]["employees_per_role"]
    )

    all_rows = []

    for role, role_config in (
        synthetic_config["roles"].items()
    ):
        employee_ids = create_employee_ids(
            role_config,
            employees_per_role,
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

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        ["timestamp", "user_id"]
    ).reset_index(drop=True)

    return df


def select_cases(
    features,
    experiment_config,
    seed,
):
    """
    Select 30 independent anomaly user-days from the
    current replication.

    Same allocation and eligibility logic as the
    original select_experiment_cases.py.
    """

    design = experiment_config["experiment_design"]
    targets = experiment_config["scenario_targets"]

    allocation = design[
        "scenario_allocation_per_role"
    ]

    rng = np.random.default_rng(seed)

    df = features.copy()

    selected_rows = []

    for role in sorted(df["role"].unique()):
        role_df = df[
            df["role"] == role
        ].copy()

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

    if len(cases) != 30:
        raise ValueError(
            f"Expected 30 selected anomaly cases, "
            f"found {len(cases)}."
        )

    if cases.duplicated(KEY_COLUMNS).any():
        raise ValueError(
            "Duplicate anomaly user-day selected."
        )

    return cases


def inject_cases(
    base_df,
    cases,
    experiment_config,
    difficulty,
    seed,
):
    """
    Apply the original project's anomaly injection logic
    entirely in memory.
    """

    rng = np.random.default_rng(seed)

    df = base_df.copy()

    ground_truth_rows = []

    for _, case in cases.iterrows():
        role = case["role"]
        scenario = case["scenario"]
        user_id = case["user_id"]
        date_value = case["date"]

        target_config = (
            experiment_config["scenario_targets"]
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

        else:
            raise ValueError(
                f"Unknown scenario: {scenario}"
            )

        ground_truth_rows.append(
            {
                "case_id": case["case_id"],
                "user_id": user_id,
                "date": pd.to_datetime(
                    date_value
                ).strftime("%Y-%m-%d"),
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

    return df, ground_truth


def normalize_keys(df):
    df = df.copy()

    df["user_id"] = (
        df["user_id"]
        .astype(str)
        .str.strip()
    )

    df["role"] = (
        df["role"]
        .astype(str)
        .str.strip()
    )

    df["date"] = (
        pd.to_datetime(df["date"])
        .dt.strftime("%Y-%m-%d")
    )

    return df


def attach_ground_truth(
    combined,
    ground_truth,
):
    """
    Add true_label only after detector inference.

    This preserves separation between ground truth and
    detector inputs.
    """

    evaluated = normalize_keys(combined)

    if ground_truth.empty:
        evaluated["true_label"] = 0
        return evaluated

    gt = normalize_keys(ground_truth)

    gt_keys = gt[
        KEY_COLUMNS
    ].drop_duplicates().copy()

    gt_keys["true_label"] = 1

    evaluated = evaluated.merge(
        gt_keys,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    evaluated["true_label"] = (
        evaluated["true_label"]
        .fillna(0)
        .astype(int)
    )

    return evaluated


def safe_divide(numerator, denominator):
    if denominator == 0:
        return np.nan

    return numerator / denominator


def calculate_metrics(
    evaluated,
    prediction_column,
):
    y_true = evaluated["true_label"].astype(int)
    y_pred = evaluated[
        prediction_column
    ].astype(int)

    tp = int(
        ((y_true == 1) & (y_pred == 1)).sum()
    )
    tn = int(
        ((y_true == 0) & (y_pred == 0)).sum()
    )
    fp = int(
        ((y_true == 0) & (y_pred == 1)).sum()
    )
    fn = int(
        ((y_true == 1) & (y_pred == 0)).sum()
    )

    total = len(evaluated)

    accuracy = safe_divide(
        tp + tn,
        total,
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    if (
        np.isnan(precision)
        or np.isnan(recall)
        or (precision + recall == 0)
    ):
        f1 = np.nan
    else:
        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    fpr = safe_divide(
        fp,
        fp + tn,
    )

    specificity = safe_divide(
        tn,
        tn + fp,
    )

    return {
        "n_user_days": total,
        "n_anomalies": int(y_true.sum()),
        "n_normal": int((y_true == 0).sum()),
        "flagged": int(y_pred.sum()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "specificity": specificity,
    }


def build_metric_row(
    condition,
    difficulty,
    replication,
    data_seed,
    case_seed,
    injection_seed,
    detector,
    metrics,
    raw_event_count,
    candidate_count,
):
    row = {
        "condition": condition,
        "difficulty": (
            difficulty
            if difficulty is not None
            else "none"
        ),
        "replication": replication,
        "data_seed": data_seed,
        "case_seed": (
            case_seed
            if case_seed is not None
            else np.nan
        ),
        "injection_seed": (
            injection_seed
            if injection_seed is not None
            else np.nan
        ),
        "detector": detector,
        "raw_event_count": raw_event_count,
        "candidate_count": candidate_count,
    }

    row.update(metrics)

    return row


def run_replication(
    condition,
    replication,
    synthetic_config,
    experiment_config,
):
    condition_info = CONDITIONS[condition]

    difficulty = condition_info["difficulty"]
    base_seed = condition_info["seed_base"]

    # Three deterministic but separate random streams.
    data_seed = base_seed + replication

    case_seed = (
        base_seed
        + 1000
        + replication
    )

    injection_seed = (
        base_seed
        + 2000
        + replication
    )

    raw_normal = generate_normal_test(
        seed=data_seed,
        synthetic_config=synthetic_config,
        experiment_config=experiment_config,
    )

    # First pipeline pass is needed for case eligibility.
    # For control, this is also the final detector pass.
    normal_result = run_raw_pipeline(
        raw_normal
    )

    if len(normal_result["features"]) != 300:
        raise ValueError(
            f"{condition} replication {replication}: "
            f"expected 300 user-days, found "
            f"{len(normal_result['features'])}."
        )

    if condition == "control":
        raw_test = raw_normal

        result = normal_result

        ground_truth = pd.DataFrame(
            columns=KEY_COLUMNS
        )

        case_seed_for_output = None
        injection_seed_for_output = None

    else:
        cases = select_cases(
            features=normal_result["features"],
            experiment_config=experiment_config,
            seed=case_seed,
        )

        raw_test, ground_truth = inject_cases(
            base_df=raw_normal,
            cases=cases,
            experiment_config=experiment_config,
            difficulty=difficulty,
            seed=injection_seed,
        )

        result = run_raw_pipeline(
            raw_test
        )

        case_seed_for_output = case_seed
        injection_seed_for_output = injection_seed

    combined = result["combined"]

    if len(combined) != 300:
        raise ValueError(
            f"{condition} replication {replication}: "
            f"expected 300 evaluated user-days, "
            f"found {len(combined)}."
        )

    evaluated = attach_ground_truth(
        combined=combined,
        ground_truth=ground_truth,
    )

    expected_anomalies = (
        0
        if condition == "control"
        else 30
    )

    actual_anomalies = int(
        evaluated["true_label"].sum()
    )

    if actual_anomalies != expected_anomalies:
        raise ValueError(
            f"{condition} replication {replication}: "
            f"expected {expected_anomalies} anomalies, "
            f"found {actual_anomalies}."
        )

    candidate_count = int(
        combined["candidate"].sum()
    )

    ml_metrics = calculate_metrics(
        evaluated,
        "prediction",
    )

    rule_metrics = calculate_metrics(
        evaluated,
        "rule_prediction",
    )

    rows = [
        build_metric_row(
            condition=condition,
            difficulty=difficulty,
            replication=replication,
            data_seed=data_seed,
            case_seed=case_seed_for_output,
            injection_seed=injection_seed_for_output,
            detector="ml",
            metrics=ml_metrics,
            raw_event_count=len(raw_test),
            candidate_count=candidate_count,
        ),
        build_metric_row(
            condition=condition,
            difficulty=difficulty,
            replication=replication,
            data_seed=data_seed,
            case_seed=case_seed_for_output,
            injection_seed=injection_seed_for_output,
            detector="rules",
            metrics=rule_metrics,
            raw_event_count=len(raw_test),
            candidate_count=candidate_count,
        ),
    ]

    case_audit = None

    if condition != "control":
        case_audit = ground_truth.copy()
        case_audit.insert(
            0,
            "replication",
            replication,
        )
        case_audit.insert(
            0,
            "condition",
            condition,
        )

    return rows, case_audit


def load_existing_metrics():
    if not METRICS_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        METRICS_PATH
    )


def replication_complete(
    existing,
    condition,
    replication,
):
    if existing.empty:
        return False

    subset = existing[
        (existing["condition"] == condition)
        & (
            existing["replication"]
            == replication
        )
    ]

    detectors = set(
        subset["detector"].astype(str)
    )

    return detectors == {
        "ml",
        "rules",
    }


def append_rows(rows):
    new_df = pd.DataFrame(rows)

    if METRICS_PATH.exists():
        new_df.to_csv(
            METRICS_PATH,
            mode="a",
            header=False,
            index=False,
        )
    else:
        new_df.to_csv(
            METRICS_PATH,
            index=False,
        )


def append_case_audit(case_audit):
    if case_audit is None:
        return

    if CASE_AUDIT_PATH.exists():
        case_audit.to_csv(
            CASE_AUDIT_PATH,
            mode="a",
            header=False,
            index=False,
        )
    else:
        case_audit.to_csv(
            CASE_AUDIT_PATH,
            index=False,
        )


def validate_final_output(
    expected_replications,
    selected_conditions,
):
    metrics = pd.read_csv(
        METRICS_PATH
    )

    problems = []

    for condition in selected_conditions:
        subset = metrics[
            metrics["condition"] == condition
        ]

        unique_reps = subset[
            "replication"
        ].nunique()

        expected_rows = (
            expected_replications * 2
        )

        if unique_reps != expected_replications:
            problems.append(
                f"{condition}: expected "
                f"{expected_replications} replications, "
                f"found {unique_reps}"
            )

        if len(subset) != expected_rows:
            problems.append(
                f"{condition}: expected "
                f"{expected_rows} metric rows, "
                f"found {len(subset)}"
            )

    if problems:
        print("\n=== VALIDATION ISSUES ===")
        for problem in problems:
            print(f"- {problem}")

        return False

    print("\n=== STAGE 30D.1 VALIDATION ===")

    for condition in selected_conditions:
        print(
            f"{condition.upper():10s}: "
            f"{expected_replications}/"
            f"{expected_replications} PASS"
        )

    total_runs = (
        expected_replications
        * len(selected_conditions)
    )

    print(
        f"\nTOTAL DATASETS: "
        f"{total_runs}/{total_runs} PASS"
    )

    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run independent robustness replications "
            "using frozen project models, thresholds, "
            "rules and anomaly definitions."
        )
    )

    parser.add_argument(
        "--replications",
        type=int,
        default=50,
        help="Replications per condition (default: 50).",
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=list(CONDITIONS.keys()),
        default=list(CONDITIONS.keys()),
        help="Conditions to run.",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Delete existing robustness outputs "
            "before running."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.fresh:
        for path in [
            METRICS_PATH,
            CASE_AUDIT_PATH,
        ]:
            if path.exists():
                path.unlink()

    synthetic_config = load_json(
        SYNTHETIC_CONFIG_PATH
    )

    experiment_config = load_json(
        EXPERIMENT_CONFIG_PATH
    )

    existing = load_existing_metrics()

    total_requested = (
        args.replications
        * len(args.conditions)
    )

    completed_now = 0
    skipped = 0

    print("=== Stage 30D.1 Robustness Replications ===")
    print(
        f"Conditions       : "
        f"{', '.join(args.conditions)}"
    )
    print(
        f"Replications     : "
        f"{args.replications} per condition"
    )
    print(
        f"Datasets planned : "
        f"{total_requested}"
    )
    print(
        "Frozen models    : YES"
    )
    print(
        "Frozen thresholds: YES"
    )
    print(
        "LLM API calls    : NO"
    )
    print()

    for condition in args.conditions:
        for replication in range(
            1,
            args.replications + 1,
        ):
            existing = load_existing_metrics()

            if replication_complete(
                existing,
                condition,
                replication,
            ):
                print(
                    f"[SKIP] {condition} "
                    f"{replication:02d}/"
                    f"{args.replications}"
                )
                skipped += 1
                continue

            print(
                f"[RUN ] {condition} "
                f"{replication:02d}/"
                f"{args.replications}"
            )

            rows, case_audit = run_replication(
                condition=condition,
                replication=replication,
                synthetic_config=synthetic_config,
                experiment_config=experiment_config,
            )

            append_rows(rows)
            append_case_audit(
                case_audit
            )

            completed_now += 1

            ml = next(
                row
                for row in rows
                if row["detector"] == "ml"
            )

            rules = next(
                row
                for row in rows
                if row["detector"] == "rules"
            )

            print(
                f"       user-days="
                f"{ml['n_user_days']} | "
                f"anomalies="
                f"{ml['n_anomalies']} | "
                f"ML flagged="
                f"{ml['flagged']} | "
                f"Rules flagged="
                f"{rules['flagged']}"
            )

    print()
    print(
        f"Completed now : {completed_now}"
    )
    print(
        f"Skipped       : {skipped}"
    )
    print(
        f"Metrics       : {METRICS_PATH}"
    )
    print(
        f"Case audit    : {CASE_AUDIT_PATH}"
    )

    validate_final_output(
        expected_replications=args.replications,
        selected_conditions=args.conditions,
    )


if __name__ == "__main__":
    main()