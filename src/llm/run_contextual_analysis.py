import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROMPT_PATH = PROJECT_ROOT / "configs" / "llm_prompt.txt"
CONTEXT_DIR = PROJECT_ROOT / "outputs" / "llm_context"
RESULT_DIR = PROJECT_ROOT / "outputs" / "llm_results"

MODEL = "gpt-5-nano-2025-08-07"
PROMPT_VERSION = "v2"

VALID_RISK_LEVELS = {"Low", "Medium", "High"}
VALID_PRIORITIES = {"Routine", "Review", "Escalate"}

EXPECTED_PRIORITY = {
    "Low": "Routine",
    "Medium": "Review",
    "High": "Escalate",
}

REQUIRED_FIELDS = {
    "risk_level",
    "investigation_priority",
    "summary",
    "evidence",
    "role_context",
    "uncertainty",
    "recommended_action",
}


def load_prompt():
    return PROMPT_PATH.read_text(
        encoding="utf-8"
    ).strip()


def load_context(experiment):
    path = CONTEXT_DIR / f"{experiment}_context.json"

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_assessment(data):

    if not isinstance(data, dict):
        raise ValueError("Output must be a JSON object.")

    fields = set(data.keys())

    missing = REQUIRED_FIELDS - fields
    extra = fields - REQUIRED_FIELDS

    if missing:
        raise ValueError(
            f"Missing fields: {sorted(missing)}"
        )

    if extra:
        raise ValueError(
            f"Unexpected fields: {sorted(extra)}"
        )

    risk = data["risk_level"]
    priority = data["investigation_priority"]

    if risk not in VALID_RISK_LEVELS:
        raise ValueError(
            f"Invalid risk_level: {risk}"
        )

    if priority not in VALID_PRIORITIES:
        raise ValueError(
            f"Invalid investigation_priority: {priority}"
        )

    if priority != EXPECTED_PRIORITY[risk]:
        raise ValueError(
            "Risk and investigation priority inconsistent."
        )

    if not isinstance(data["evidence"], list):
        raise ValueError(
            "evidence must be a list."
        )

    if not 2 <= len(data["evidence"]) <= 4:
        raise ValueError(
            "evidence must contain 2 to 4 items."
        )

    for item in data["evidence"]:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                "Every evidence item must be non-empty text."
            )

    for field in [
        "summary",
        "role_context",
        "uncertainty",
        "recommended_action",
    ]:
        if (
            not isinstance(data[field], str)
            or not data[field].strip()
        ):
            raise ValueError(
                f"{field} must be non-empty text."
            )


def call_llm(client, prompt, candidate):

    candidate_json = json.dumps(
        candidate,
        indent=2,
        ensure_ascii=False,
    )

    response = client.responses.create(
        model=MODEL,
        instructions=prompt,
        input=(
            "Assess the following synthetic banking "
            "employee user-day.\n\n"
            f"{candidate_json}"
        ),
    )

    raw_output = response.output_text.strip()

    if not raw_output:
        raise ValueError(
            "OpenAI returned empty output."
        )

    assessment = json.loads(raw_output)

    validate_assessment(assessment)

    usage = getattr(response, "usage", None)

    usage_data = {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }

    if usage is not None:
        usage_data = {
            "input_tokens": getattr(
                usage, "input_tokens", None
            ),
            "output_tokens": getattr(
                usage, "output_tokens", None
            ),
            "total_tokens": getattr(
                usage, "total_tokens", None
            ),
        }

    return {
        "response_id": response.id,
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "assessment": assessment,
        "usage": usage_data,
    }


def candidate_key(candidate):

    context = candidate["employee_context"]

    return (
        f"{context['user_id']}|"
        f"{context['date']}|"
        f"{context['role']}"
    )


def load_existing_results(path):

    if not path.exists():
        return {
            "model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "results": [],
        }

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if data.get("model") != MODEL:
        raise ValueError(
            "Existing batch file uses a different model."
        )

    if data.get("prompt_version") != PROMPT_VERSION:
        raise ValueError(
            "Existing batch file uses a different prompt."
        )

    return data


def save_results(path, data):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(".tmp")

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


def run_batch(experiment):

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY was not found."
        )

    client = OpenAI(
        api_key=api_key
    )

    prompt = load_prompt()
    context = load_context(experiment)

    candidates = context["candidates"]

    output_path = (
        RESULT_DIR
        / f"{experiment}_llm_results.json"
    )

    batch = load_existing_results(
        output_path
    )

    batch["experiment"] = experiment
    batch["model"] = MODEL
    batch["prompt_version"] = PROMPT_VERSION
    batch["candidate_count"] = len(candidates)

    completed = {
        result["candidate_key"]
        for result in batch["results"]
    }

    print("=== LLM Batch Analysis ===")
    print(f"Experiment       : {experiment}")
    print(f"Model            : {MODEL}")
    print(f"Prompt version   : {PROMPT_VERSION}")
    print(f"Total candidates : {len(candidates)}")
    print(f"Already complete : {len(completed)}")
    print()

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):

        key = candidate_key(candidate)

        if key in completed:
            print(
                f"[{index}/{len(candidates)}] "
                f"SKIP {key}"
            )
            continue

        print(
            f"[{index}/{len(candidates)}] "
            f"Calling OpenAI: {key}"
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

                record = {
                    "candidate_key": key,
                    "candidate_input": candidate,
                    **result,
                }

                batch["results"].append(
                    record
                )

                completed.add(key)

                save_results(
                    output_path,
                    batch,
                )

                assessment = result[
                    "assessment"
                ]

                print(
                    "  -> "
                    f"{assessment['risk_level']} / "
                    f"{assessment['investigation_priority']}"
                )

                break

            except Exception as exc:

                print(
                    f"  Attempt {attempt}/"
                    f"{max_attempts} failed: {exc}"
                )

                if attempt == max_attempts:
                    print(
                        "\nBatch stopped safely."
                    )
                    print(
                        "Completed results have been saved."
                    )
                    raise

                time.sleep(
                    2 ** attempt
                )

    # ------------------------------------------
    # Final summary
    # ------------------------------------------

    risk_counts = {
        "Low": 0,
        "Medium": 0,
        "High": 0,
    }

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    for result in batch["results"]:

        risk = result[
            "assessment"
        ]["risk_level"]

        risk_counts[risk] += 1

        usage = result.get(
            "usage"
        ) or {}

        total_input_tokens += (
            usage.get("input_tokens") or 0
        )

        total_output_tokens += (
            usage.get("output_tokens") or 0
        )

        total_tokens += (
            usage.get("total_tokens") or 0
        )

    batch["summary"] = {
        "completed": len(
            batch["results"]
        ),
        "risk_counts": risk_counts,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
    }

    save_results(
        output_path,
        batch,
    )

    print("\n=== Batch Summary ===")
    print(
        f"Completed : "
        f"{len(batch['results'])}/"
        f"{len(candidates)}"
    )

    print(
        f"Low       : {risk_counts['Low']}"
    )
    print(
        f"Medium    : {risk_counts['Medium']}"
    )
    print(
        f"High      : {risk_counts['High']}"
    )

    print(
        f"Tokens    : {total_tokens}"
    )

    print(
        f"\nSaved: {output_path}"
    )

    print("\nBatch: PASS")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        required=True,
        choices=[
            "experiment_1",
            "experiment_2",
            "experiment_3",
        ],
    )

    parser.add_argument(
        "--batch",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.batch:
        raise ValueError(
            "Use --batch to run the frozen LLM experiment."
        )

    run_batch(
        args.experiment
    )


if __name__ == "__main__":
    main()