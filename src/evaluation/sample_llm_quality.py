import json
import random
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LLM_DIR = PROJECT_ROOT / "outputs" / "llm_results"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

EXPERIMENTS = [
    "experiment_1",
    "experiment_2",
    "experiment_3",
]

RANDOM_SEED = 2026


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def flatten_result(experiment, result):

    candidate = result["candidate_input"]
    assessment = result["assessment"]

    employee = candidate["employee_context"]

    return {
        "experiment": experiment,
        "candidate_key": result["candidate_key"],
        "user_id": employee["user_id"],
        "date": employee["date"],
        "role": employee["role"],

        "risk_level":
            assessment["risk_level"],

        "investigation_priority":
            assessment["investigation_priority"],

        "summary":
            assessment["summary"],

        "evidence":
            " | ".join(
                assessment["evidence"]
            ),

        "role_context":
            assessment["role_context"],

        "uncertainty":
            assessment["uncertainty"],

        "recommended_action":
            assessment["recommended_action"],

        "candidate_input_json":
            json.dumps(
                candidate,
                ensure_ascii=False,
            ),
    }


def main():

    random.seed(RANDOM_SEED)

    all_rows = []

    for experiment in EXPERIMENTS:

        path = (
            LLM_DIR
            / f"{experiment}_llm_results.json"
        )

        data = load_json(path)

        for result in data["results"]:

            all_rows.append(
                flatten_result(
                    experiment,
                    result,
                )
            )

    df = pd.DataFrame(all_rows)

    samples = []

    # --------------------------------------------------
    # 1. Include every High assessment.
    # --------------------------------------------------

    high = df[
        df["risk_level"] == "High"
    ].copy()

    high["sample_reason"] = "all_high"

    samples.append(high)

    # --------------------------------------------------
    # 2. Sample Medium cases from every experiment.
    #    Five per experiment.
    # --------------------------------------------------

    for experiment in EXPERIMENTS:

        medium = df[
            (df["experiment"] == experiment)
            &
            (df["risk_level"] == "Medium")
        ].copy()

        n = min(5, len(medium))

        sampled = medium.sample(
            n=n,
            random_state=(
                RANDOM_SEED
                + EXPERIMENTS.index(experiment)
            ),
        )

        sampled["sample_reason"] = (
            "medium_random_sample"
        )

        samples.append(sampled)

    audit = pd.concat(
        samples,
        ignore_index=True,
    )

    # Remove duplicates defensively.
    audit = audit.drop_duplicates(
        subset=[
            "experiment",
            "candidate_key",
        ]
    ).copy()

    # --------------------------------------------------
    # Manual audit fields
    # --------------------------------------------------

    audit["grounding_pass"] = ""
    audit["role_awareness_pass"] = ""
    audit["unsupported_claims_pass"] = ""
    audit["uncertainty_pass"] = ""

    audit["manual_notes"] = ""

    output_path = (
        METRICS_DIR
        / "llm_quality_manual_audit.csv"
    )

    audit.to_csv(
        output_path,
        index=False,
    )

    print("=" * 70)
    print("LLM QUALITY AUDIT SAMPLE")
    print("=" * 70)

    print(
        f"Total LLM assessments : {len(df)}"
    )

    print(
        f"All High included      : {len(high)}"
    )

    print(
        f"Medium sampled         : "
        f"{len(audit) - len(high)}"
    )

    print(
        f"Total manual sample    : {len(audit)}"
    )

    print("\nSample by experiment/risk:")

    print(
        audit.groupby(
            [
                "experiment",
                "risk_level",
            ]
        )
        .size()
        .to_string()
    )

    print(
        "\nSaved:"
        f"\n{output_path}"
    )

    print(
        "\nNext step: manually review "
        "the sampled assessments."
    )


if __name__ == "__main__":
    main()