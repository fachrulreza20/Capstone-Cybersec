import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LLM_RESULT_DIR = PROJECT_ROOT / "outputs" / "llm_results"
GROUND_TRUTH_DIR = PROJECT_ROOT / "data" / "ground_truth"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics"

EXPERIMENTS = [
    "experiment_1",
    "experiment_2",
    "experiment_3",
]

EXPECTED_COUNTS = {
    "experiment_1": 76,
    "experiment_2": 76,
    "experiment_3": 72,
}

EXPECTED_PRIORITY = {
    "Low": "Routine",
    "Medium": "Review",
    "High": "Escalate",
}

REQUIRED_ASSESSMENT_FIELDS = {
    "risk_level",
    "investigation_priority",
    "summary",
    "evidence",
    "role_context",
    "uncertainty",
    "recommended_action",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def find_ground_truth_file(experiment):
    candidates = list(
        GROUND_TRUTH_DIR.glob(f"*{experiment}*.csv")
    )

    if not candidates:
        candidates = list(
            GROUND_TRUTH_DIR.glob(
                f"*{experiment.replace('_', '')}*.csv"
            )
        )

    if not candidates:
        raise FileNotFoundError(
            f"Could not find ground truth CSV for {experiment} "
            f"in {GROUND_TRUTH_DIR}"
        )

    if len(candidates) > 1:
        print(
            f"WARNING: multiple ground truth files found "
            f"for {experiment}:"
        )
        for path in candidates:
            print(f"  {path.name}")
        print(f"Using: {candidates[0].name}")

    return candidates[0]


def validate_llm_file(experiment, data):
    errors = []

    if data.get("model") != "gpt-5-nano-2025-08-07":
        errors.append(
            f"Unexpected model: {data.get('model')}"
        )

    if data.get("prompt_version") != "v2":
        errors.append(
            f"Unexpected prompt version: "
            f"{data.get('prompt_version')}"
        )

    results = data.get("results")

    if not isinstance(results, list):
        errors.append("results is not a list.")
        return errors

    expected = EXPECTED_COUNTS[experiment]

    if len(results) != expected:
        errors.append(
            f"Expected {expected} results, found {len(results)}."
        )

    keys = [
        result.get("candidate_key")
        for result in results
    ]

    if None in keys:
        errors.append(
            "At least one result has no candidate_key."
        )

    if len(keys) != len(set(keys)):
        errors.append(
            "Duplicate candidate_key detected."
        )

    for index, result in enumerate(results, start=1):

        assessment = result.get("assessment")

        if not isinstance(assessment, dict):
            errors.append(
                f"Result {index}: assessment missing/not dict."
            )
            continue

        fields = set(assessment.keys())

        if fields != REQUIRED_ASSESSMENT_FIELDS:
            errors.append(
                f"Result {index}: assessment fields invalid."
            )

        risk = assessment.get("risk_level")
        priority = assessment.get(
            "investigation_priority"
        )

        if risk not in EXPECTED_PRIORITY:
            errors.append(
                f"Result {index}: invalid risk {risk}."
            )

        elif priority != EXPECTED_PRIORITY[risk]:
            errors.append(
                f"Result {index}: risk/priority mismatch."
            )

        evidence = assessment.get("evidence")

        if not isinstance(evidence, list):
            errors.append(
                f"Result {index}: evidence not list."
            )

        elif not 2 <= len(evidence) <= 4:
            errors.append(
                f"Result {index}: evidence count "
                f"{len(evidence)}."
            )

        candidate = result.get("candidate_input")

        if not isinstance(candidate, dict):
            errors.append(
                f"Result {index}: candidate_input missing."
            )
            continue

        candidate_text = json.dumps(
            candidate
        ).lower()

        forbidden = [
            "true_label",
            "ground_truth",
            "case_id",
            "scenario",
            "difficulty",
            "experiment_difficulty",
        ]

        for field in forbidden:
            if field in candidate_text:
                errors.append(
                    f"Result {index}: possible leakage "
                    f"field '{field}'."
                )

    return errors


def extract_llm_rows(experiment, data):
    rows = []

    for result in data["results"]:

        candidate = result["candidate_input"]
        employee = candidate["employee_context"]
        assessment = result["assessment"]

        risk = assessment["risk_level"]

        rows.append(
            {
                "experiment": experiment,
                "user_id": employee["user_id"],
                "date": employee["date"],
                "role": employee["role"],
                "candidate_key": result[
                    "candidate_key"
                ],
                "risk_level": risk,
                "investigation_priority":
                    assessment[
                        "investigation_priority"
                    ],
                "llm_investigate":
                    0 if risk == "Low" else 1,
            }
        )

    return pd.DataFrame(rows)


def normalise_ground_truth(gt):
    print(
        "Ground truth columns:",
        list(gt.columns)
    )

    required_identity = {
        "user_id",
        "date",
        "role",
    }

    missing = required_identity - set(gt.columns)

    if missing:
        raise ValueError(
            f"Ground truth missing identity columns: "
            f"{sorted(missing)}"
        )

    gt["date"] = gt["date"].astype(str)

    # Ground-truth files contain the injected anomaly cases.
    # Therefore every row in these files is positive ground truth.
    gt["true_label"] = 1

    return gt


def confusion_metrics(y_true, y_pred):
    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred).astype(int)

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

    total = tp + tn + fp + fn

    accuracy = (
        (tp + tn) / total
        if total else 0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) else 0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) else 0
    )

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
    }


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_llm = []
    integrity_rows = []

    print("=" * 70)
    print("LLM RESULT INTEGRITY CHECK")
    print("=" * 70)

    for experiment in EXPERIMENTS:

        path = (
            LLM_RESULT_DIR
            / f"{experiment}_llm_results.json"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing: {path}"
            )

        data = load_json(path)

        errors = validate_llm_file(
            experiment,
            data,
        )

        result_count = len(
            data.get("results", [])
        )

        integrity_rows.append(
            {
                "experiment": experiment,
                "expected": EXPECTED_COUNTS[
                    experiment
                ],
                "actual": result_count,
                "errors": len(errors),
                "status":
                    "PASS"
                    if not errors
                    else "FAIL",
            }
        )

        print(f"\n{experiment}")
        print(
            f"  Results : {result_count}/"
            f"{EXPECTED_COUNTS[experiment]}"
        )

        if errors:
            print("  Status  : FAIL")
            for error in errors:
                print(f"    - {error}")
        else:
            print("  Status  : PASS")

        if not errors:
            all_llm.append(
                extract_llm_rows(
                    experiment,
                    data,
                )
            )

    integrity_df = pd.DataFrame(
        integrity_rows
    )

    integrity_df.to_csv(
        OUTPUT_DIR
        / "llm_integrity_check.csv",
        index=False,
    )

    if (
        integrity_df["status"] != "PASS"
    ).any():

        print(
            "\nSTOP: integrity check failed."
        )
        print(
            "Do not continue evaluation yet."
        )
        return

    llm_df = pd.concat(
        all_llm,
        ignore_index=True,
    )
    
    
    
    
    

    print("\n" + "=" * 70)
    print("LLM DISTRIBUTION")
    print("=" * 70)

    distribution = (
        llm_df.groupby(
            [
                "experiment",
                "risk_level",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    for column in [
        "Low",
        "Medium",
        "High",
    ]:
        if column not in distribution.columns:
            distribution[column] = 0

    distribution = distribution[
        ["Low", "Medium", "High"]
    ]

    print(distribution)

    distribution.to_csv(
        OUTPUT_DIR
        / "llm_risk_distribution.csv"
    )

    # --------------------------------------------------
    # Candidate-level evaluation
    # --------------------------------------------------

    candidate_metric_rows = []

    print("\n" + "=" * 70)
    print("CANDIDATE-LEVEL EVALUATION")
    print("=" * 70)

    for experiment in EXPERIMENTS:

        exp_llm = llm_df[
            llm_df["experiment"]
            == experiment
        ].copy()

        gt_path = find_ground_truth_file(
            experiment
        )

        gt = pd.read_csv(gt_path)

        gt = normalise_ground_truth(gt)

        gt_keys = set(
            zip(
                gt["user_id"],
                gt["date"],
                gt["role"],
            )
        )

        exp_llm["true_label"] = [
            1
            if (
                row.user_id,
                str(row.date),
                row.role,
            )
            in gt_keys
            else 0
            for row in exp_llm.itertuples()
        ]

        metrics = confusion_metrics(
            exp_llm["true_label"],
            exp_llm["llm_investigate"],
        )

        metrics["experiment"] = experiment
        metrics["scope"] = "candidate_only"

        candidate_metric_rows.append(
            metrics
        )

        print(f"\n{experiment}")
        print(
            f"  Candidate count : "
            f"{len(exp_llm)}"
        )
        print(
            f"  True anomaly    : "
            f"{exp_llm['true_label'].sum()}"
        )
        print(
            f"  Normal candidate: "
            f"{(exp_llm['true_label'] == 0).sum()}"
        )
        print(
            f"  TP/TN/FP/FN     : "
            f"{metrics['TP']}/"
            f"{metrics['TN']}/"
            f"{metrics['FP']}/"
            f"{metrics['FN']}"
        )
        print(
            f"  Precision       : "
            f"{metrics['precision']:.4f}"
        )
        print(
            f"  Recall          : "
            f"{metrics['recall']:.4f}"
        )

        joined_path = (
            OUTPUT_DIR
            / f"{experiment}_llm_joined.csv"
        )

        exp_llm.to_csv(
            joined_path,
            index=False,
        )

    candidate_metrics_df = pd.DataFrame(
        candidate_metric_rows
    )

    candidate_metrics_df.to_csv(
        OUTPUT_DIR
        / "llm_candidate_metrics.csv",
        index=False,
    )

    # --------------------------------------------------
    # Risk distribution by ground-truth class
    # --------------------------------------------------

    joined_frames = []

    for experiment in EXPERIMENTS:

        path = (
            OUTPUT_DIR
            / f"{experiment}_llm_joined.csv"
        )

        joined_frames.append(
            pd.read_csv(path)
        )

    joined_all = pd.concat(
        joined_frames,
        ignore_index=True,
    )

    class_distribution = (
        joined_all.groupby(
            [
                "experiment",
                "true_label",
                "risk_level",
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    class_distribution.to_csv(
        OUTPUT_DIR
        / "llm_risk_by_ground_truth.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("RISK BY GROUND-TRUTH CLASS")
    print("=" * 70)

    print(
        class_distribution.to_string(
            index=False
        )
    )


    # --------------------------------------------------
    # High-risk prioritisation analysis
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("HIGH-RISK PRIORITISATION")
    print("=" * 70)

    high_rows = []

    for experiment in EXPERIMENTS:

        exp = joined_all[
            joined_all["experiment"] == experiment
        ].copy()

        high = exp[
            exp["risk_level"] == "High"
        ]

        high_total = len(high)
        high_true = int(high["true_label"].sum())
        high_normal = high_total - high_true

        precision_high = (
            high_true / high_total
            if high_total > 0
            else 0
        )

        high_rows.append(
            {
                "experiment": experiment,
                "high_total": high_total,
                "high_true_anomaly": high_true,
                "high_normal": high_normal,
                "high_precision": precision_high,
            }
        )

        print(f"\n{experiment}")
        print(f"  High total       : {high_total}")
        print(f"  Injected anomaly : {high_true}")
        print(f"  Normal candidate : {high_normal}")
        print(
            f"  High precision   : "
            f"{precision_high:.4f}"
        )

    high_df = pd.DataFrame(high_rows)

    high_df.to_csv(
        OUTPUT_DIR / "llm_high_prioritisation.csv",
        index=False,
    )

    # --------------------------------------------------
    # Join scenario and difficulty information
    # --------------------------------------------------

    enriched_frames = []

    for experiment in EXPERIMENTS:

        joined = joined_all[
            joined_all["experiment"] == experiment
        ].copy()

        gt_path = find_ground_truth_file(experiment)
        gt = pd.read_csv(gt_path)

        gt["date"] = gt["date"].astype(str)
        joined["date"] = joined["date"].astype(str)

        gt_extra = gt[
            [
                "user_id",
                "date",
                "role",
                "case_id",
                "scenario",
                "difficulty",
            ]
        ].copy()

        enriched = joined.merge(
            gt_extra,
            on=["user_id", "date", "role"],
            how="left",
        )

        enriched_frames.append(enriched)

    enriched_all = pd.concat(
        enriched_frames,
        ignore_index=True,
    )

    # --------------------------------------------------
    # Risk by scenario
    # --------------------------------------------------

    anomaly_only = enriched_all[
        enriched_all["true_label"] == 1
    ].copy()

    scenario_risk = (
        anomaly_only.groupby(
            [
                "experiment",
                "scenario",
                "risk_level",
            ]
        )
        .size()
        .reset_index(name="count")
    )

    scenario_risk.to_csv(
        OUTPUT_DIR / "llm_risk_by_scenario.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("RISK BY INJECTED SCENARIO")
    print("=" * 70)

    print(
        scenario_risk.to_string(index=False)
    )

    # --------------------------------------------------
    # Risk by role
    # --------------------------------------------------

    role_risk = (
        enriched_all.groupby(
            [
                "experiment",
                "role",
                "risk_level",
            ]
        )
        .size()
        .reset_index(name="count")
    )

    role_risk.to_csv(
        OUTPUT_DIR / "llm_risk_by_role.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("RISK BY ROLE")
    print("=" * 70)

    print(
        role_risk.to_string(index=False)
    )

    # --------------------------------------------------
    # Matched anomaly case analysis
    # --------------------------------------------------

    risk_score = {
        "Low": 0,
        "Medium": 1,
        "High": 2,
    }

    anomaly_only["risk_score"] = (
        anomaly_only["risk_level"]
        .map(risk_score)
    )

    matched = anomaly_only[
        anomaly_only["case_id"].notna()
    ].copy()

    matched_pivot = matched.pivot_table(
        index=[
            "case_id",
            "user_id",
            "role",
            "scenario",
        ],
        columns="experiment",
        values="risk_score",
        aggfunc="first",
    ).reset_index()

    matched_pivot.to_csv(
        OUTPUT_DIR
        / "llm_matched_case_risk.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("MATCHED CASE RISK ANALYSIS")
    print("=" * 70)

    expected_exp_columns = [
        "experiment_1",
        "experiment_2",
        "experiment_3",
    ]

    complete_cases = matched_pivot.dropna(
        subset=expected_exp_columns
    ).copy()

    print(
        f"Cases available in all 3 experiments: "
        f"{len(complete_cases)}"
    )

    if len(complete_cases) > 0:

        non_increasing = (
            (
                complete_cases["experiment_1"]
                >= complete_cases["experiment_2"]
            )
            &
            (
                complete_cases["experiment_2"]
                >= complete_cases["experiment_3"]
            )
        )

        decreased = (
            (
                complete_cases["experiment_1"]
                > complete_cases["experiment_2"]
            )
            |
            (
                complete_cases["experiment_2"]
                > complete_cases["experiment_3"]
            )
        )

        increased = (
            (
                complete_cases["experiment_1"]
                < complete_cases["experiment_2"]
            )
            |
            (
                complete_cases["experiment_2"]
                < complete_cases["experiment_3"]
            )
        )

        print(
            f"Non-increasing risk across difficulty: "
            f"{int(non_increasing.sum())}/"
            f"{len(complete_cases)}"
        )

        print(
            f"At least one risk decrease: "
            f"{int(decreased.sum())}/"
            f"{len(complete_cases)}"
        )

        print(
            f"At least one risk increase: "
            f"{int(increased.sum())}/"
            f"{len(complete_cases)}"
        )

    # --------------------------------------------------
    # End-to-end LLM-assisted evaluation
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("END-TO-END LLM-ASSISTED EVALUATION")
    print("=" * 70)

    end_to_end_rows = []

    for experiment in EXPERIMENTS:

        exp = joined_all[
            joined_all["experiment"] == experiment
        ]

        candidate_tp = int(
            (
                (exp["true_label"] == 1)
                &
                (exp["llm_investigate"] == 1)
            ).sum()
        )

        candidate_fp = int(
            (
                (exp["true_label"] == 0)
                &
                (exp["llm_investigate"] == 1)
            ).sum()
        )

        # Each experiment contains:
        # 30 injected anomaly user-days
        # 270 normal user-days
        total_anomalies = 30
        total_normals = 270

        tp = candidate_tp
        fp = candidate_fp

        fn = total_anomalies - tp
        tn = total_normals - fp

        metrics = confusion_metrics(
            [1] * tp
            + [0] * tn
            + [0] * fp
            + [1] * fn,
            [1] * tp
            + [0] * tn
            + [1] * fp
            + [0] * fn,
        )

        metrics["experiment"] = experiment
        metrics["scope"] = "end_to_end"

        end_to_end_rows.append(metrics)

        print(f"\n{experiment}")
        print(
            f"  TP/TN/FP/FN : "
            f"{metrics['TP']}/"
            f"{metrics['TN']}/"
            f"{metrics['FP']}/"
            f"{metrics['FN']}"
        )
        print(
            f"  Accuracy     : "
            f"{metrics['accuracy']:.4f}"
        )
        print(
            f"  Precision    : "
            f"{metrics['precision']:.4f}"
        )
        print(
            f"  Recall       : "
            f"{metrics['recall']:.4f}"
        )
        print(
            f"  F1           : "
            f"{metrics['f1']:.4f}"
        )
        print(
            f"  FPR          : "
            f"{metrics['fpr']:.4f}"
        )

    end_to_end_df = pd.DataFrame(
        end_to_end_rows
    )

    end_to_end_df.to_csv(
        OUTPUT_DIR
        / "llm_end_to_end_metrics.csv",
        index=False,
    )



    print("\n" + "=" * 70)
    print("FINAL")
    print("=" * 70)

    print(
        f"Total LLM assessments: "
        f"{len(llm_df)}"
    )

    print(
        "\nSaved evaluation evidence to:"
    )

    print(
        "  outputs/metrics/"
        "llm_integrity_check.csv"
    )
    print(
        "  outputs/metrics/"
        "llm_risk_distribution.csv"
    )
    print(
        "  outputs/metrics/"
        "llm_candidate_metrics.csv"
    )
    print(
        "  outputs/metrics/"
        "llm_risk_by_ground_truth.csv"
    )

    print(
        "\nLLM evaluation stage: PASS"
    )


if __name__ == "__main__":
    main()