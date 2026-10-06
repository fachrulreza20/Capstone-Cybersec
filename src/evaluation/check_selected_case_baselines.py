import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_config.json"

CASES_PATH = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "experiment_cases.csv"
)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    config = load_config()
    cases = pd.read_csv(CASES_PATH)

    results = []

    for _, row in cases.iterrows():
        role = row["role"]
        scenario = row["scenario"]

        target = (
            config["scenario_targets"]
            [role][scenario]["subtle"]
        )

        compatible = True
        checks = []

        if scenario == "unusual_export":
            lower = target[0]

            value = row["download_total"]

            passed = value < lower
            compatible &= passed

            checks.append(
                f"download {value} < {lower}: {passed}"
            )

        elif scenario == "unusual_record_access":
            lower = target[0]

            value = row["records_accessed_total"]

            passed = value < lower
            compatible &= passed

            checks.append(
                f"records {value} < {lower}: {passed}"
            )

        elif scenario == "temporal_combined":
            for feature in [
                "last_activity_hour",
                "download_total",
                "records_accessed_total",
            ]:
                lower = target[feature][0]
                value = row[feature]

                passed = value < lower
                compatible &= passed

                checks.append(
                    f"{feature} {value:.2f} < "
                    f"{lower}: {passed}"
                )

        results.append(
            {
                "case_id": row["case_id"],
                "user_id": row["user_id"],
                "date": row["date"],
                "role": role,
                "scenario": scenario,
                "compatible": compatible,
                "check_details": " | ".join(checks),
            }
        )

    result_df = pd.DataFrame(results)

    print("=== Selected Case Baseline Check ===")
    print(f"Total cases : {len(result_df)}")

    passed = result_df["compatible"].sum()
    failed = (~result_df["compatible"]).sum()

    print(f"Compatible  : {passed}")
    print(f"Incompatible: {failed}")

    if failed > 0:
        print("\n=== Incompatible Cases ===")

        print(
            result_df[
                ~result_df["compatible"]
            ][
                [
                    "case_id",
                    "role",
                    "scenario",
                    "check_details",
                ]
            ].to_string(index=False)
        )

        print(
            "\nRESULT: CASE SELECTION NEEDS ADJUSTMENT"
        )

    else:
        print(
            "\nRESULT: ALL CASES ARE COMPATIBLE"
        )
        print(
            "Safe to proceed to anomaly injection."
        )


if __name__ == "__main__":
    main()