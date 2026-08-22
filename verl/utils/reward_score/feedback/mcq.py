from __future__ import annotations

import re
from typing import Sequence

from data.utils.mcq import DEFAULT_MCQ_ALLOWED_CHOICES


def extract_xml_answer(text: str) -> str:
    matches = list(
        re.finditer(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    )
    if not matches:
        return ""
    return matches[-1].group(1).strip()


def is_correct_format(
    text: str, allowed_choices: Sequence[str] = DEFAULT_MCQ_ALLOWED_CHOICES
) -> bool:
    allowed_pattern = "|".join(re.escape(choice.upper()) for choice in allowed_choices)
    pattern = rf"<answer>\s*({allowed_pattern})\s*</answer>\s*$"
    return re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) is not None


def _extract_last_boxed_answer(text: str) -> str | None:
    matches = list(re.finditer(r"\\boxed\{([^{}]+)\}", text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _extract_xml_choice(
    text: str, allowed_choices: Sequence[str] = DEFAULT_MCQ_ALLOWED_CHOICES
) -> str | None:
    candidate = extract_xml_answer(text).upper().rstrip(".").rstrip("/")
    if candidate in {choice.upper() for choice in allowed_choices}:
        return candidate
    return None


def extract_multiple_choice_answer(
    solution: str, allowed_choices: Sequence[str]
) -> str | None:
    normalized_allowed_choices = tuple(choice.upper() for choice in allowed_choices)

    boxed_answer = _extract_last_boxed_answer(solution)
    if boxed_answer is not None:
        candidate = boxed_answer.strip().upper().rstrip(".").rstrip("/")
        if candidate in normalized_allowed_choices:
            return candidate

    xml_answer = _extract_xml_choice(solution, normalized_allowed_choices)
    if xml_answer is not None:
        return xml_answer

    patterns = [
        r"(?:final answer|correct answer|the answer)\s*(?:is|:)\s*\(?([A-Z])\)?",
        r"answer is\s*\(?([A-Z])\)?",
        r"answer:\s*\(?([A-Z])\)?",
        r"answer\s*\(?([A-Z])\)?",
        r"\(([A-Z])\)",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, solution, flags=re.IGNORECASE))
        if matches:
            candidate = matches[-1].group(1).upper()
            if candidate in normalized_allowed_choices:
                return candidate

    fallback_pattern = (
        r"\b(" + "|".join(re.escape(choice) for choice in normalized_allowed_choices) + r")\b"
    )
    fallback_matches = re.findall(fallback_pattern, solution.upper())
    if fallback_matches:
        return fallback_matches[-1]

    return None


def evaluate_mcq_response(
    solution: str, ground_truth: str, allowed_choices: Sequence[str]
) -> dict[str, float | str | None]:
    pred = extract_multiple_choice_answer(solution=solution, allowed_choices=allowed_choices)
    normalized_ground_truth = ground_truth.upper()
    acc = float(pred == normalized_ground_truth)
    incorrect_format = float(pred is None)
    return {
        "acc": acc,
        "incorrect_format": incorrect_format,
        "pred": pred,
    }


def flatten_benchmark_metrics(
    processed_metrics: dict[str, dict[str, dict[str, float]]],
) -> dict[str, float]:
    metric_dict: dict[str, float] = {}

    for data_source, var2metric2val in processed_metrics.items():
        for var_name in ("acc", "incorrect_format", "truncated", "response_length"):
            metric2val = var2metric2val.get(var_name, {})
            for metric_name, metric_val in metric2val.items():
                metric_dict[f"benchmark/{data_source}/{var_name}/{metric_name}"] = metric_val

        acc_metrics = var2metric2val.get("acc", {})
        if acc_metrics:
            n_max = max(int(name.split("@")[-1].split("/")[0]) for name in acc_metrics.keys())
            mean_key = f"mean@{n_max}"
            majority_key = f"maj@{n_max}/mean"

            if mean_key in acc_metrics:
                metric_dict[f"benchmark/{data_source}/acc"] = acc_metrics[mean_key]
            if majority_key in acc_metrics:
                metric_dict[f"benchmark/{data_source}/acc_majority"] = acc_metrics[majority_key]

            for summary_var in ("incorrect_format", "truncated", "response_length"):
                summary_metrics = var2metric2val.get(summary_var, {})
                if mean_key in summary_metrics:
                    metric_dict[f"benchmark/{data_source}/{summary_var}"] = summary_metrics[mean_key]

    return metric_dict


def compute_score(
    solution: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> dict:
    extra_info = extra_info or {}
    reward_model_info = extra_info.get("reward_model_info", {})
    allowed_choices = tuple(
        reward_model_info.get("allowed_choices", DEFAULT_MCQ_ALLOWED_CHOICES)
    )
    answer_extraction = reward_model_info.get("answer_extraction", "xml")
    normalized_ground_truth = ground_truth.strip().upper()
    if answer_extraction == "flexible":
        pred = extract_multiple_choice_answer(solution, allowed_choices)
    else:
        pred = _extract_xml_choice(solution, allowed_choices)
    reward = float(pred == normalized_ground_truth)
    if answer_extraction == "flexible":
        incorrect_format = float(pred is None)
    else:
        incorrect_format = float(not is_correct_format(solution, allowed_choices))

    return {
        "score": reward,
        "acc": reward,
        "pred": pred or "",
        "incorrect_format": incorrect_format,
        "feedback": "",
    }
