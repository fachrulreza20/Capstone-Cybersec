import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.llm.run_contextual_analysis import (
    MODEL,
    PROMPT_VERSION,
    call_llm,
    load_prompt,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONTEXT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "robustness"
    / "llm_control_context.json"
)

AUDIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "robustness"
    / "llm_control_sample.csv"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "robustness"
    / "llm_control_results.json"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "robustness"
    / "llm_control_summary.csv"
)


VALID_GROUPS = {
    "flagged_normal",
    "unflagged_normal",
}

VALID_RISK_LEVELS = {
    "Low",
    "Medium",
    "High",
}


def load_context():
    if not CONTEXT_PATH.exists():
        raise FileNotFoundError(
            f"Context not found: {CONTEXT_PATH}"
        )

    with open(
        CONTEXT_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if data.get("sample_size") != 60:
        raise ValueError(
            "Expected exactly 60 control cases."
        )

    cases = data.get("cases", [])

    if len(cases) != 60:
        raise ValueError(
            f"Expected 60 cases, found {len(cases)}."
        )

    return data


def load_audit():
    if not AUDIT_PATH.exists():
        raise FileNotFoundError(
            f"Audit sample not found: {AUDIT_PATH}"
        )

    audit = pd.read_csv(
        AUDIT_PATH
    )

    if len(audit) != 60:
        raise ValueError(
            f"Expected 60 audit rows, "
            f"found {len(audit)}."
        )

    if audit["control_case_id"].duplicated().any():
        raise ValueError(
            "Duplicate control_case_id found."
        )

    groups = set(
        audit["control_group"].astype(str)
    )

    if groups != VALID_GROUPS:
        raise ValueError(
            f"Unexpected control groups: {groups}"
        )

    if not (
        audit["true_label"].astype(int) == 0
    ).all():
        raise ValueError(
            "Control audit contains non-normal "
            "ground truth."
        )

    return audit


def load_existing_results():
    if not RESULT_PATH.exists():
        return {
            "study": "llm_normal_control_validation",
            "model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "results": [],
        }

    with open(
        RESULT_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if data.get("model") != MODEL:
        raise ValueError(
            "Existing result file uses "
            "a different model."
        )

    if (
        data.get("prompt_version")
        != PROMPT_VERSION
    ):
        raise ValueError(
            "Existing result file uses "
            "a different prompt version."
        )

    return data


def save_json_atomic(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        ".tmp"
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temp_path.replace(path)


def validate_context_against_audit(
    context,
    audit,
):
    context_ids = {
        case["control_case_id"]
        for case in context["cases"]
    }

    audit_ids = set(
        audit["control_case_id"].astype(str)
    )

    if context_ids != audit_ids:
        raise ValueError(
            "Context/audit case IDs do not match."
        )

    # Critical leakage check.
    forbidden = {
        "true_label",
        "control_group",
        "replication",
        "data_seed",
        "scenario",
        "difficulty",
        "case_id",
    }

    def recurse(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in forbidden:
                    raise ValueError(
                        "Forbidden field leaked into "
                        f"candidate input: {key}"
                    )
                recurse(child)

        elif isinstance(value, list):
            for child in value:
                recurse(child)

    for case in context["cases"]:
        recurse(
            case["candidate_input"]
        )


def preview_study(
    context,
    audit,
):
    print(
        "=== Stage 30D.3B "
        "LLM Normal-Control Validation ==="
    )
    print()
    print(f"Model          : {MODEL}")
    print(
        f"Prompt version : {PROMPT_VERSION}"
    )
    print(
        f"Cases          : {len(context['cases'])}"
    )
    print(
        "Ground truth   : "
        "60 synthetic normal user-days"
    )
    print(
        "LLM receives GT: NO"
    )
    print()

    table = (
        audit.groupby(
            ["control_group", "role"]
        )
        .size()
        .unstack(fill_value=0)
    )

    print(table.to_string())
    print()

    print(
        "Flagged normal   : "
        f"{(audit['control_group'] == 'flagged_normal').sum()}"
    )
    print(
        "Unflagged normal : "
        f"{(audit['control_group'] == 'unflagged_normal').sum()}"
    )


def run_calls(
    context,
    audit,
):
    load_dotenv()

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY was not found."
        )

    client = OpenAI(
        api_key=api_key
    )

    prompt = load_prompt()

    batch = load_existing_results()

    batch["study"] = (
        "llm_normal_control_validation"
    )
    batch["model"] = MODEL
    batch["prompt_version"] = PROMPT_VERSION
    batch["sample_size"] = 60

    completed = {
        result["control_case_id"]
        for result in batch["results"]
    }

    audit_lookup = (
        audit.set_index(
            "control_case_id"
        )
        .to_dict(
            orient="index"
        )
    )

    print()
    print("=== API EXECUTION ===")
    print(
        f"Already complete : {len(completed)}"
    )
    print(
        f"Remaining        : "
        f"{60 - len(completed)}"
    )
    print()

    for index, case in enumerate(
        context["cases"],
        start=1,
    ):
        case_id = case[
            "control_case_id"
        ]

        candidate = case[
            "candidate_input"
        ]

        if case_id in completed:
            print(
                f"[{index:02d}/60] "
                f"SKIP {case_id}"
            )
            continue

        metadata = audit_lookup[
            case_id
        ]

        print(
            f"[{index:02d}/60] "
            f"Calling OpenAI: {case_id}"
        )

        max_attempts = 3

        for attempt in range(
            1,
            max_attempts + 1,
        ):
            try:
                result = call_llm(
                    client,
                    prompt,
                    candidate,
                )

                assessment = result[
                    "assessment"
                ]

                record = {
                    "control_case_id": case_id,

                    # Evaluation metadata is attached
                    # AFTER the API response.
                    "control_group": metadata[
                        "control_group"
                    ],
                    "role": metadata[
                        "role"
                    ],
                    "replication": int(
                        metadata[
                            "replication"
                        ]
                    ),
                    "data_seed": int(
                        metadata[
                            "data_seed"
                        ]
                    ),
                    "true_label": 0,

                    "candidate_input": candidate,

                    **result,
                }

                batch[
                    "results"
                ].append(
                    record
                )

                completed.add(
                    case_id
                )

                save_json_atomic(
                    RESULT_PATH,
                    batch,
                )

                print(
                    "  -> "
                    f"{assessment['risk_level']} / "
                    f"{assessment['investigation_priority']}"
                )

                break

            except Exception as exc:
                print(
                    f"  Attempt "
                    f"{attempt}/{max_attempts} "
                    f"failed: {exc}"
                )

                if attempt == max_attempts:
                    print(
                        "\nBatch stopped safely."
                    )
                    print(
                        "Completed results "
                        "have been saved."
                    )
                    raise

                time.sleep(
                    2 ** attempt
                )

    return batch


def build_summary(batch):
    rows = []

    for result in batch["results"]:
        assessment = result[
            "assessment"
        ]

        rows.append(
            {
                "control_case_id": result[
                    "control_case_id"
                ],
                "control_group": result[
                    "control_group"
                ],
                "role": result[
                    "role"
                ],
                "risk_level": assessment[
                    "risk_level"
                ],
                "investigation_priority": (
                    assessment[
                        "investigation_priority"
                    ]
                ),
                "input_tokens": (
                    result.get(
                        "usage", {}
                    ).get(
                        "input_tokens"
                    )
                ),
                "output_tokens": (
                    result.get(
                        "usage", {}
                    ).get(
                        "output_tokens"
                    )
                ),
                "total_tokens": (
                    result.get(
                        "usage", {}
                    ).get(
                        "total_tokens"
                    )
                ),
            }
        )

    results_df = pd.DataFrame(
        rows
    )

    if len(results_df) != 60:
        raise ValueError(
            f"Expected 60 completed LLM results, "
            f"found {len(results_df)}."
        )

    summary_rows = []

    for group in [
        "flagged_normal",
        "unflagged_normal",
    ]:
        subset = results_df[
            results_df["control_group"]
            == group
        ]

        if len(subset) != 30:
            raise ValueError(
                f"{group}: expected 30 results, "
                f"found {len(subset)}."
            )

        for risk in [
            "Low",
            "Medium",
            "High",
        ]:
            count = int(
                (
                    subset["risk_level"]
                    == risk
                ).sum()
            )

            summary_rows.append(
                {
                    "control_group": group,
                    "risk_level": risk,
                    "count": count,
                    "percentage": (
                        count
                        / len(subset)
                        * 100
                    ),
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    return results_df, summary


def print_summary(
    results_df,
    summary,
):
    print()
    print(
        "=== LLM CONTROL RESULTS ==="
    )

    for group in [
        "flagged_normal",
        "unflagged_normal",
    ]:
        print()
        print(
            f"--- {group.upper()} ---"
        )

        subset = summary[
            summary["control_group"]
            == group
        ]

        for _, row in subset.iterrows():
            print(
                f"{row['risk_level']:6s}: "
                f"{int(row['count']):2d}/30 "
                f"({row['percentage']:.1f}%)"
            )

    print()
    print("=== BY ROLE ===")

    role_table = pd.crosstab(
        [
            results_df[
                "control_group"
            ],
            results_df["role"],
        ],
        results_df["risk_level"],
    )

    print(
        role_table.to_string()
    )

    total_tokens = (
        pd.to_numeric(
            results_df["total_tokens"],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    print()
    print(
        f"Total API calls : "
        f"{len(results_df)}"
    )
    print(
        f"Total tokens    : "
        f"{int(total_tokens)}"
    )

    print()
    print(
        f"Results : {RESULT_PATH}"
    )
    print(
        f"Summary : {SUMMARY_PATH}"
    )

    print()
    print(
        "STAGE 30D.3B: PASS"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Actually perform the 60 "
            "OpenAI API calls."
        ),
    )

    args = parser.parse_args()

    context = load_context()
    audit = load_audit()

    validate_context_against_audit(
        context,
        audit,
    )

    preview_study(
        context,
        audit,
    )

    if not args.execute:
        print()
        print(
            "DRY RUN ONLY — no API calls made."
        )
        print(
            "Validation: PASS"
        )
        print()
        print(
            "To execute the frozen 60-call "
            "control study:"
        )
        print(
            "python -m "
            "src.robustness.run_llm_control "
            "--execute"
        )
        return

    batch = run_calls(
        context,
        audit,
    )

    results_df, summary = (
        build_summary(
            batch
        )
    )

    print_summary(
        results_df,
        summary,
    )


if __name__ == "__main__":
    main()