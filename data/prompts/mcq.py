from __future__ import annotations

from typing import Sequence

from data.utils.mcq import DEFAULT_MCQ_ALLOWED_CHOICES

GPQA_BASELINE_SYSTEM_PROMPT = (
    "You are a very intelligent assistant, who follows instructions directly."
)

MMLU_PRO_SYSTEM_PROMPT = (
    "You are an knowledge expert, you are supposed to answer the multi-choice question "
    "to derive your final answer as `The answer is ...`."
)


def build_gpqa_zero_shot_prompt(problem: str, options: dict[str, str]) -> str:
    choices = "\n".join(
        f"({letter}) {options[letter]}" for letter in DEFAULT_MCQ_ALLOWED_CHOICES
    )
    return (
        f"What is the correct answer to this question: {problem}\n\n"
        f"Choices:\n{choices}\n\n"
        'Format your response as follows: "The correct answer is (insert answer here)"'
    )


def build_medical_zero_shot_prompt(question: str, options: dict[str, str]) -> str:
    choices = "\n".join(
        f"({letter}) {options[letter]}" for letter in DEFAULT_MCQ_ALLOWED_CHOICES
    )
    return (
        f"What is the correct answer to this medical question: {question}\n\n"
        f"Choices:\n{choices}\n\n"
        'Format your response as follows: "The correct answer is (insert answer here)"'
    )


def format_mmlu_subject(subject: str) -> str:
    return " ".join(subject.split("_"))


def format_mmlu_question(question: str, options: Sequence[str]) -> str:
    choices = "".join(
        f" {letter}. {option}\n"
        for letter, option in zip(DEFAULT_MCQ_ALLOWED_CHOICES, options, strict=True)
    )
    return f"{question.strip()}\n{choices}Answer:"


def build_mmlu_zero_shot_prompt(
    subject: str, question: str, options: Sequence[str]
) -> str:
    description = (
        "The following are multiple choice questions (with answers) about "
        f"{format_mmlu_subject(subject)}."
    )
    return f"{description}\n\n{format_mmlu_question(question, options)}"


def format_mmlu_pro_options(options: Sequence[str]) -> str:
    allowed_choices = tuple(chr(ord("A") + i) for i in range(len(options)))
    option_lines = "\n".join(
        f"({letter}): {option}"
        for letter, option in zip(allowed_choices, options, strict=True)
    )
    return f"Options are:\n{option_lines}"


def build_mmlu_pro_zero_shot_prompt(question: str, options: Sequence[str]) -> str:
    return f"Q: {question}\n{format_mmlu_pro_options(options)}\n"

