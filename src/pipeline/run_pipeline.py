import argparse
import json
from pathlib import Path

import pandas as pd

from src.pipeline.detection_pipeline import (
    run_raw_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "pipeline"
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the stable banking behavioural "
            "anomaly-detection pipeline."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to raw synthetic banking "
            "audit-log CSV."
        ),
    )

    parser.add_argument(
        "--name",
        default="pipeline_run",
        help="Output run name.",
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    ).resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    print(
        "=== Banking Behavioural "
        "Anomaly Detection Pipeline ==="
    )

    print(
        f"\nInput: {input_path}"
    )

    raw_df = pd.read_csv(
        input_path,
        parse_dates=["timestamp"],
    )

    print(
        f"Raw events: {len(raw_df):,}"
    )

    result = run_raw_pipeline(
        raw_df
    )

    features = result["features"]
    combined = result["combined"]
    analyst_table = result[
        "analyst_table"
    ]
    packages = result[
        "candidate_packages"
    ]

    ml_count = int(
        combined["prediction"].sum()
    )

    rule_count = int(
        combined["rule_prediction"].sum()
    )

    candidate_count = int(
        combined["candidate"].sum()
    )

    print(
        f"User-days: {len(features):,}"
    )

    print(
        f"ML anomalies: {ml_count:,}"
    )

    print(
        f"Rule anomalies: {rule_count:,}"
    )

    print(
        f"Investigation candidates: "
        f"{candidate_count:,}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    analyst_path = (
        OUTPUT_DIR
        / f"{args.name}_analyst_table.csv"
    )

    context_path = (
        OUTPUT_DIR
        / f"{args.name}_llm_context.json"
    )

    analyst_table.to_csv(
        analyst_path,
        index=False,
    )

    context_output = {
        "candidate_selection": (
            "Isolation Forest anomaly OR "
            "deterministic rule anomaly"
        ),
        "input_design": (
            "Structured user-day behavioural "
            "context. Raw event timeline excluded."
        ),
        "candidate_count": len(packages),
        "candidates": packages,
    }

    with open(
        context_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            context_output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nAnalyst table: {analyst_path}"
    )

    print(
        f"LLM context  : {context_path}"
    )

    print(
        "\nGround truth used: NO"
    )

    print(
        "OpenAI API called : NO"
    )

    print(
        "\nPipeline: PASS"
    )


if __name__ == "__main__":
    main()