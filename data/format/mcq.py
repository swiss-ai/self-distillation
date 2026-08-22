from __future__ import annotations

import random
from typing import Any

from datasets import load_dataset

from data.prompts.mcq import (
    GPQA_BASELINE_SYSTEM_PROMPT,
    MMLU_PRO_SYSTEM_PROMPT,
    build_gpqa_zero_shot_prompt,
    build_medical_zero_shot_prompt,
    build_mmlu_pro_zero_shot_prompt,
    build_mmlu_zero_shot_prompt,
)
from data.utils.mcq import (
    DEFAULT_MCQ_ALLOWED_CHOICES,
    MMLU_SUBJECT_TO_DOMAIN,
    get_mmlu_metric_paths,
    normalize_choice_value,
    normalize_max_samples,
    normalize_mmlu_pro_category,
    normalize_mmlu_subject,
)

SUPPORTED_MCQ_BENCHMARKS = frozenset(
    {"gpqa_diamond", "medmcqa", "medqa_en", "mmlu", "mmlu_pro"}
)


def _build_prompt_messages(
    prompt: str, *, system_prompt: str | None = None
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _resolve_max_samples(benchmark_config, global_max_samples: int) -> int:
    return normalize_max_samples(benchmark_config.get("max_samples", global_max_samples))


def load_gpqa_diamond_examples(
    benchmark_config, global_max_samples: int
) -> list[dict[str, Any]]:
    dataset_name = benchmark_config.get("dataset_name", "Idavidrein/gpqa")
    subset = benchmark_config.get("subset", "gpqa_diamond")
    split = benchmark_config.get("split", "train")
    category = benchmark_config.get("category", None)
    seed = int(benchmark_config.get("seed", 42))

    dataset = load_dataset(dataset_name, subset, split=split)
    if category is not None:
        dataset = dataset.filter(lambda row: row["High-level domain"] == category)

    max_samples = _resolve_max_samples(benchmark_config, global_max_samples)
    if max_samples == 0:
        return []
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    examples = []
    for example_idx, example in enumerate(dataset):
        choices = [
            example["Correct Answer"],
            example["Incorrect Answer 1"],
            example["Incorrect Answer 2"],
            example["Incorrect Answer 3"],
        ]
        rng = random.Random(seed + example_idx)
        rng.shuffle(choices)

        options_to_answers = {
            letter: answer
            for letter, answer in zip(
                DEFAULT_MCQ_ALLOWED_CHOICES, choices, strict=True
            )
        }
        ground_truth = next(
            letter
            for letter, answer in options_to_answers.items()
            if answer == example["Correct Answer"]
        )

        examples.append(
            {
                "uid": f"gpqa_diamond:{example_idx}",
                "prompt": _build_prompt_messages(
                    build_gpqa_zero_shot_prompt(
                        problem=example["Question"],
                        options=options_to_answers,
                    ),
                    system_prompt=GPQA_BASELINE_SYSTEM_PROMPT,
                ),
                "extra_info": {"data_source": "mcq/gpqa_diamond", "metric_paths": ()},
                "reward_model": {
                    "ground_truth": ground_truth,
                    "scoring_module": "mcq",
                    "info": {
                        "allowed_choices": DEFAULT_MCQ_ALLOWED_CHOICES,
                        "answer_extraction": "flexible",
                    },
                },
                "answer": "",
            }
        )

    return examples


def load_medmcqa_examples(
    benchmark_config, global_max_samples: int
) -> list[dict[str, Any]]:
    dataset_name = benchmark_config.get("dataset_name", "openlifescienceai/medmcqa")
    split = benchmark_config.get("split", "validation")
    subject = benchmark_config.get("subject", None)

    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.filter(lambda row: row.get("choice_type", "single") == "single")
    if subject is not None:
        normalized_subject = str(subject).strip().lower()
        dataset = dataset.filter(
            lambda row: row["subject_name"].strip().lower() == normalized_subject
        )

    max_samples = _resolve_max_samples(benchmark_config, global_max_samples)
    if max_samples == 0:
        return []
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    examples = []
    for example_idx, example in enumerate(dataset):
        options = {
            "A": example["opa"],
            "B": example["opb"],
            "C": example["opc"],
            "D": example["opd"],
        }
        examples.append(
            {
                "uid": f"medmcqa:{example_idx}",
                "prompt": _build_prompt_messages(
                    build_medical_zero_shot_prompt(
                        question=example["question"],
                        options=options,
                    ),
                    system_prompt=GPQA_BASELINE_SYSTEM_PROMPT,
                ),
                "extra_info": {"data_source": "mcq/medmcqa", "metric_paths": ()},
                "reward_model": {
                    "ground_truth": DEFAULT_MCQ_ALLOWED_CHOICES[int(example["cop"]) - 1],
                    "scoring_module": "mcq",
                    "info": {
                        "allowed_choices": DEFAULT_MCQ_ALLOWED_CHOICES,
                        "answer_extraction": "flexible",
                    },
                },
                "answer": "",
            }
        )

    return examples


def load_medqa_examples(
    benchmark_config, global_max_samples: int
) -> list[dict[str, Any]]:
    dataset_name = benchmark_config.get("dataset_name", "openlifescienceai/medqa")
    split = benchmark_config.get("split", "test")

    dataset = load_dataset(dataset_name, split=split)

    max_samples = _resolve_max_samples(benchmark_config, global_max_samples)
    if max_samples == 0:
        return []
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    examples = []
    for example_idx, example in enumerate(dataset):
        data = example["data"]
        options = {
            "A": data["Options"]["A"],
            "B": data["Options"]["B"],
            "C": data["Options"]["C"],
            "D": data["Options"]["D"],
        }
        examples.append(
            {
                "uid": f"medqa_en:{example_idx}",
                "prompt": _build_prompt_messages(
                    build_medical_zero_shot_prompt(
                        question=data["Question"],
                        options=options,
                    ),
                    system_prompt=GPQA_BASELINE_SYSTEM_PROMPT,
                ),
                "extra_info": {"data_source": "mcq/medqa_en", "metric_paths": ()},
                "reward_model": {
                    "ground_truth": data["Correct Option"].strip().upper(),
                    "scoring_module": "mcq",
                    "info": {
                        "allowed_choices": DEFAULT_MCQ_ALLOWED_CHOICES,
                        "answer_extraction": "flexible",
                    },
                },
                "answer": "",
            }
        )

    return examples


def load_mmlu_examples(benchmark_config, global_max_samples: int) -> list[dict[str, Any]]:
    dataset_name = benchmark_config.get("dataset_name", "cais/mmlu")
    split = benchmark_config.get("split", "test")
    subject = benchmark_config.get("subject", None)

    if subject in (None, "all"):
        subjects = list(MMLU_SUBJECT_TO_DOMAIN)
    else:
        normalized_subject = normalize_mmlu_subject(str(subject))
        if normalized_subject not in MMLU_SUBJECT_TO_DOMAIN:
            raise ValueError(
                f"Unsupported MMLU subject '{subject}'. Supported subjects: "
                f"{sorted(MMLU_SUBJECT_TO_DOMAIN)} or 'all'."
            )
        subjects = [normalized_subject]

    max_samples = _resolve_max_samples(benchmark_config, global_max_samples)
    if max_samples == 0:
        return []

    examples = []
    for current_subject in subjects:
        if max_samples > 0 and len(examples) >= max_samples:
            break

        eval_dataset = load_dataset(dataset_name, current_subject, split=split)
        remaining = max_samples - len(examples) if max_samples > 0 else len(eval_dataset)
        num_examples = min(len(eval_dataset), remaining)

        for example_idx in range(num_examples):
            example = eval_dataset[example_idx]
            examples.append(
                {
                    "uid": f"mmlu:{current_subject}:{example_idx}",
                    "prompt": _build_prompt_messages(
                        build_mmlu_zero_shot_prompt(
                            subject=current_subject,
                            question=example["question"],
                            options=example["choices"],
                        )
                    ),
                    "extra_info": {
                        "data_source": "mcq/mmlu",
                        "metric_paths": get_mmlu_metric_paths(current_subject),
                    },
                    "reward_model": {
                        "ground_truth": normalize_choice_value(
                            example["answer"], DEFAULT_MCQ_ALLOWED_CHOICES
                        ),
                        "scoring_module": "mcq",
                        "info": {
                            "allowed_choices": DEFAULT_MCQ_ALLOWED_CHOICES,
                            "answer_extraction": "flexible",
                        },
                    },
                    "answer": "",
                }
            )

    return examples


def load_mmlu_pro_examples(
    benchmark_config, global_max_samples: int
) -> list[dict[str, Any]]:
    dataset_name = benchmark_config.get("dataset_name", "TIGER-Lab/MMLU-Pro")
    split = benchmark_config.get("split", "test")
    category = benchmark_config.get("category", None)

    eval_dataset = load_dataset(dataset_name, split=split)

    normalized_category = (
        None if category is None else normalize_mmlu_pro_category(str(category))
    )
    if normalized_category is not None:
        eval_dataset = eval_dataset.filter(
            lambda row: normalize_mmlu_pro_category(row["category"])
            == normalized_category
        )

    max_samples = _resolve_max_samples(benchmark_config, global_max_samples)
    if max_samples == 0:
        return []
    if max_samples > 0:
        eval_dataset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))

    examples = []
    for example_idx, example in enumerate(eval_dataset):
        allowed_choices = tuple(chr(ord("A") + i) for i in range(len(example["options"])))
        row_category = normalize_mmlu_pro_category(example["category"])
        examples.append(
            {
                "uid": f"mmlu_pro:{row_category}:{example_idx}",
                "prompt": _build_prompt_messages(
                    build_mmlu_pro_zero_shot_prompt(
                        example["question"], example["options"]
                    ),
                    system_prompt=MMLU_PRO_SYSTEM_PROMPT,
                ),
                "extra_info": {"data_source": "mcq/mmlu_pro", "metric_paths": ()},
                "reward_model": {
                    "ground_truth": normalize_choice_value(
                        example["answer"], allowed_choices
                    ),
                    "scoring_module": "mcq",
                    "info": {
                        "allowed_choices": allowed_choices,
                        "answer_extraction": "flexible",
                    },
                },
                "answer": "",
            }
        )

    return examples
