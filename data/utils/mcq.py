from __future__ import annotations

from typing import Sequence

DEFAULT_MCQ_ALLOWED_CHOICES = ("A", "B", "C", "D")

MMLU_SUBJECT_TO_DOMAIN = {
    "abstract_algebra": "stem",
    "anatomy": "other",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "stem",
    "world_religions": "humanities",
}


def normalize_max_samples(value) -> int:
    if value is None:
        return -1
    return int(value)


def coalesce_int(value, fallback: int) -> int:
    if value is None:
        return fallback
    return int(value)


def normalize_choice_value(answer: str | int, allowed_choices: Sequence[str]) -> str:
    normalized_allowed_choices = tuple(choice.upper() for choice in allowed_choices)
    if isinstance(answer, int):
        return normalized_allowed_choices[answer]

    normalized = str(answer).strip().upper()
    if normalized in normalized_allowed_choices:
        return normalized

    if len(normalized) == 1 and normalized.isalpha():
        return normalized

    raise ValueError(
        f"Unsupported answer value '{answer}' for choices {normalized_allowed_choices}"
    )


def normalize_mmlu_subject(subject: str) -> str:
    return subject.strip().lower().replace(" ", "_")


def normalize_mmlu_pro_category(category: str) -> str:
    return category.strip().lower().replace("_", " ")


def get_mmlu_domain(subject: str) -> str:
    try:
        return MMLU_SUBJECT_TO_DOMAIN[subject]
    except KeyError as exc:
        raise ValueError(f"Missing MMLU domain mapping for subject '{subject}'") from exc


def get_mmlu_metric_paths(subject: str) -> tuple[str, str, str]:
    domain = get_mmlu_domain(subject)
    return (
        "mcq/mmlu",
        f"mcq/mmlu/{domain}",
        f"mcq/mmlu/{domain}/{subject}",
    )

