import json
from pathlib import Path

import pandas as pd

from src.features.build_user_day_features import build_user_day_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_config.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
GT_DIR = PROJECT_ROOT / "data" / "ground_truth"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def within_range(value, target_range, tolerance=0.02):
    low, high = target_range
    return (low - tolerance) <= value <= (high + tolerance)


def main():
    config = load_config()

    all_passed = True

    for experiment_name, difficulty in config["difficulty_levels"].items():

        print(
            f"\n=== {experiment_name} ({difficulty}) ==="
        )

        raw_path = RAW_DIR / f"{experiment_name}_test.csv"
        gt_path = GT_DIR / f"{experiment_name}_ground_truth.csv"

        raw = pd.read_csv(raw_path)
        gt = pd.read_csv(gt_path)

        features = build_user_day_features(raw)


        # Normalise merge keys before matching ground truth
        gt["date"] = pd.to_datetime(gt["date"]).dt.strftime("%Y-%m-%d")
        features["date"] = pd.to_datetime(features["date"]).dt.strftime("%Y-%m-%d")

        gt["user_id"] = gt["user_id"].astype(str)
        features["user_id"] = features["user_id"].astype(str)

        gt["role"] = gt["role"].astype(str).str.strip()
        features["role"] = features["role"].astype(str).str.strip()





        output_path = (
            PROCESSED_DIR
            / f"{experiment_name}_user_day.csv"
        )

        features.to_csv(output_path, index=False)

        print(f"Raw events       : {len(raw):,}")
        print(f"User-day rows    : {len(features)}")
        print(f"Ground truth     : {len(gt)}")

        experiment_passed = True

        if len(features) != 300:
            print("FAIL: Expected 300 user-day rows.")
            experiment_passed = False

        if len(gt) != 30:
            print("FAIL: Expected 30 anomaly cases.")
            experiment_passed = False

        merged = gt.merge(
            features,
            on=["user_id", "date", "role"],
            how="left",
            validate="one_to_one",
        )

        missing_matches = merged["event_count"].isna().sum()

        if missing_matches > 0:
            print(
                f"FAIL: {missing_matches} ground-truth cases "
                "did not match user-day features."
            )
            experiment_passed = False





        if len(merged) != 30:
            print("FAIL: Ground truth merge problem.")
            experiment_passed = False

        failed_cases = []

        for _, row in merged.iterrows():

            role = row["role"]
            scenario = row["scenario"]

            target = (
                config["scenario_targets"]
                [role][scenario][difficulty]
            )

            passed = True

            if scenario == "unusual_export":

                passed = within_range(
                    row["download_total"],
                    target,
                )

            elif scenario == "unusual_record_access":

                passed = within_range(
                    row["records_accessed_total"],
                    target,
                )

            elif scenario == "temporal_combined":

                passed = (
                    within_range(
                        row["download_total"],
                        target["download_total"],
                    )
                    and within_range(
                        row["records_accessed_total"],
                        target["records_accessed_total"],
                    )
                    and within_range(
                        row["last_activity_hour"],
                        target["last_activity_hour"],
                    )
                )

            if not passed:
                failed_cases.append(
                    {
                        "case_id": row["case_id"],
                        "role": role,
                        "scenario": scenario,
                    }
                )

        if failed_cases:
            print(
                f"FAIL: {len(failed_cases)} cases "
                f"outside target ranges."
            )

            for case in failed_cases:
                print(case)

            experiment_passed = False

        if experiment_passed:
            print(
                "PASS: 300 user-days, "
                "30 anomaly cases, "
                "all targets valid."
            )
        else:
            all_passed = False

    print("\n=== FINAL RESULT ===")

    if all_passed:
        print(
            "PASS: All experiments passed integrity checks."
        )
    else:
        print(
            "FAIL: Fix experiment generation before ML scoring."
        )


if __name__ == "__main__":
    main()