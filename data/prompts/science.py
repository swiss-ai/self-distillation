from typing import Any, Callable, Dict, List

from data.prompts.common import (PROMPT_DATASET_COLUMN_NAMES,
                                 PROMPT_DATASET_FEATURES,
                                 make_single_turn_prompt, normalize_message)
from datasets import Dataset, concatenate_datasets, load_dataset

TEXTBOOK_REASONING_DATA_SOURCE = "MegaScience/TextbookReasoning"
NATURAL_REASONING_DATA_SOURCE = "facebook/natural_reasoning"
NEMOTRON_SCIENCE_V1_DATA_SOURCE = "nvidia/Nemotron-Science-v1"
MEDICAL_O1_VERIFIABLE_PROBLEM_DATA_SOURCE = "FreedomIntelligence/medical-o1-verifiable-problem"
MULTI_SUBJECT_RLVR = "virtuoussy/Multi-subject-RLVR"

SCIENCE_DATASETS = [
    TEXTBOOK_REASONING_DATA_SOURCE,
    NATURAL_REASONING_DATA_SOURCE,
    NEMOTRON_SCIENCE_V1_DATA_SOURCE,
    MEDICAL_O1_VERIFIABLE_PROBLEM_DATA_SOURCE,
    MULTI_SUBJECT_RLVR,
]

MULTI_SUBJECT_RLVR_VERIFIER_PROMPT_TEMPLATE = """
Given a problem, determine whether the final answer in the provided (incomplete) solution process matches the reference answer.  
The reference answer may be one single option character (e.g., A, B, C, D), a numerical value, an expression, or a list of answers if multiple questions are involved.  
**The reference answer may be in Chinese or another language, but your evaluation should be language-agnostic.**  

Your task:  
- Compare the final output of the solution process with the reference answer.  
- If they **match exactly**, output **YES**.  
- If they **do not match**, output **NO**.  
- If the solution process is unclear, incomplete, or ambiguous, assume it is incorrect and output **NO**.  

Your output must be strictly **'YES'** or **'NO'**, with no additional words, punctuation, or explanation.  

---

**Question:**  
{question}  

**Solution Process (Final Step Only):**  
{response}  

**Reference Answer:**  
{reference}  

**Output:**  
"""


MEDICAL_O1_VERIFIER_PROMPT_TEMPLATE = '''<Model Response>
{response}
</Model Response>

<Reference Answer>
{reference}
</Reference Answer>

Your task is to evaluate the model response by comparing it to the reference answer. If the model response is correct and aligns with the reference answer, output "True" . If it is incorrect or fails to select the correct option (if options are provided), output "False"'''


def load_textbook_reasoning(
    categories: List[str] = ['biology', 'chemistry', 'cs', 'economics', 'math', 'medicine', 'physics']
) -> Dataset:
    """
    Load TextbookReasoning dataset

    Rows: 652k

    Original Columns:
        question: str
            LLM-generated question grounded in textbook content
        answer: str
            LLM-generated step-by-step reasoning solution
        subject: str
            subject/category label (math, medicine, biology, physics, chemistry, cs)
        reference_answer: str
            truthful reference answers extracted from textbooks
    """

    def filter_textbook_reasoning(example) -> bool:
        return example["subject"] in categories

    def map_textbook_reasoning_example(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        return {
            "prompt": make_single_turn_prompt(example["question"]),
            "reference": {"reference_answer": example["reference_answer"]},
            "data_source": TEXTBOOK_REASONING_DATA_SOURCE,
            "meta_information": {
                "subject": example["subject"],
                "reasoning_answer": example["answer"],
            },
            "data_source_id": str(idx),
            "turn": 0,
        }
    
    ds = load_dataset(TEXTBOOK_REASONING_DATA_SOURCE, split="train")
    ds = ds.filter(filter_textbook_reasoning)
    ds = ds.map(
        map_textbook_reasoning_example, 
        with_indices=True, 
        remove_columns=ds.column_names,
        features=PROMPT_DATASET_FEATURES,
    )
    return ds.select_columns(PROMPT_DATASET_COLUMN_NAMES)


def load_natural_reasoning() -> Dataset:
    """
    Load natural_reasoning dataset

    Rows: 1,145,824

    Original columns:
        question: str
        reference_answer: str
        responses: list of {response_model: str, response: str} entries
    """

    def map_natural_reasoning_example(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        reference_answer = example.get("reference_answer", "")
        return {
            "prompt": make_single_turn_prompt(example["question"]),
            "reference": {"reference_answer": reference_answer} if reference_answer else {},
            "data_source": NATURAL_REASONING_DATA_SOURCE,
            "meta_information": {
                "responses": example["responses"],
            },
            "data_source_id": str(idx),
            "turn": 0,
        }

    ds = load_dataset(NATURAL_REASONING_DATA_SOURCE, split="train")
    ds = ds.map(
        map_natural_reasoning_example,
        with_indices=True,
        remove_columns=ds.column_names,
        features=PROMPT_DATASET_FEATURES,
    )
    return ds.select_columns(PROMPT_DATASET_COLUMN_NAMES)


def load_nemotron_science_v1() -> Dataset:
    """
    Load Nemotron-Science-v1 dataset

    Rows: 
        MCQ: 174k
        RQA: 52.2k

    Original columns:
        uuid: str
        messages: list (user question and assistant response pair; in messages format)
        license: str
        used_in: list[str]
        tools: list
    """

    def map_nemotron_science_example(
        example: Dict[str, Any], idx: int, subset_name: str
    ) -> Dict[str, Any]:
        messages = example["messages"]
        if len(messages) != 2:
            raise ValueError(
                f"Expected exactly 2 messages for {subset_name}/{example['uuid']}, got {len(messages)}"
            )

        user_message, assistant_message = messages
        if user_message.get("role") != "user" or assistant_message.get("role") != "assistant":
            raise ValueError(
                f"Expected [user, assistant] message order for {subset_name}/{example['uuid']}, "
                f"got {[message.get('role') for message in messages]}"
            )

        return {
            "prompt": [normalize_message(user_message)],
            "reference": {},
            "data_source": NEMOTRON_SCIENCE_V1_DATA_SOURCE,
            "meta_information": {
                "subset": subset_name,
                "responses": [normalize_message(assistant_message)],
                "license": example.get("license"),
                "used_in": example.get("used_in", []),
                "tools": example.get("tools", []),
            },
            "data_source_id": str(example["uuid"]),
            "turn": 0,
        }

    datasets = []
    for subset_name in ("MCQ", "RQA"):
        subset = load_dataset(NEMOTRON_SCIENCE_V1_DATA_SOURCE, split=subset_name)
        subset = subset.map(
            lambda example, idx, subset_name=subset_name: map_nemotron_science_example(
                example, idx, subset_name
            ),
            with_indices=True,
            remove_columns=subset.column_names,
            features=PROMPT_DATASET_FEATURES,
        )
        datasets.append(subset.select_columns(PROMPT_DATASET_COLUMN_NAMES))

    return concatenate_datasets(datasets)


def load_medical_o1_verifiable_problem() -> Dataset:
    """
    Load medical-o1-verifiable-problem dataset

    Rows: 40,644

    Original columns:
        Open-ended Verifiable Question: str
        Ground-True Answer: str

    Unified Columns:
        prompt: json
            single-turn chat prompt in messages format
        reference: json
            ground-truth answer for verifiable evaluation
        data_source: str
            Hugging Face dataset name
        meta_information: json
            dataset-specific metadata
        data_source_id: int
            original row number
        turn: int
            assistant turn index, starting at 0
    """

    def map_medical_o1_example(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        return {
            "prompt": make_single_turn_prompt(example["Open-ended Verifiable Question"]),
            "reference": {
                "ground_truth_answer": example["Ground-True Answer"],
                "verifier_model_path": "FreedomIntelligence/medical_o1_verifier_3B",
                "verifier_prompt_template": MEDICAL_O1_VERIFIER_PROMPT_TEMPLATE,
            },
            "data_source": MEDICAL_O1_VERIFIABLE_PROBLEM_DATA_SOURCE,
            "meta_information": {"subject": "medicine"},
            "data_source_id": str(idx),
            "turn": 0,
        }

    ds = load_dataset(MEDICAL_O1_VERIFIABLE_PROBLEM_DATA_SOURCE, split="train")
    ds = ds.map(
        map_medical_o1_example,
        with_indices=True,
        remove_columns=ds.column_names,
        features=PROMPT_DATASET_FEATURES,
    )
    return ds.select_columns(PROMPT_DATASET_COLUMN_NAMES)

def load_multi_subject_rlvr() -> Dataset:
    """
    Load Multi-subject-RLVR dataset (ExamQA transformed into freeform QA)

    Rows:
        train: 573k

    Original columns:
        query: list (system prompt, user prompt)
        label: str
        subject: string (always none)
        subset: string (always none)
    """

    def map_multi_subject_rlvr_example(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        query = example["query"]
        if len(query) != 2:
            raise ValueError(f"Expected exactly 2 query messages for row {idx}, got {len(query)}")

        system_message, user_message = query
        if system_message.get("role") != "system" or user_message.get("role") != "user":
            raise ValueError(
                f"Expected [system, user] query order for row {idx}, got {[message.get('role') for message in query]}"
            )

        return {
            "prompt": [
                normalize_message(system_message),
                normalize_message(user_message),
            ],
            "reference": {
                "ground_truth_answer": example["label"],
                "verifier_model_path": "virtuoussy/Qwen2.5-7B-Instruct-RLVR",
                "verifier_prompt_template": MULTI_SUBJECT_RLVR_VERIFIER_PROMPT_TEMPLATE,
            },
            "data_source": MULTI_SUBJECT_RLVR,
            "meta_information": {},
            "data_source_id": str(idx),
            "turn": 0,
        }

    ds = load_dataset(MULTI_SUBJECT_RLVR, split="train")
    ds = ds.map(
        map_multi_subject_rlvr_example,
        with_indices=True,
        remove_columns=ds.column_names,
        features=PROMPT_DATASET_FEATURES,
    )
    return ds.select_columns(PROMPT_DATASET_COLUMN_NAMES)


def load_science() -> Dataset:
    datasets = [
        load_textbook_reasoning(
            categories=['biology', 'chemistry', 'cs', 'economics', 'medicine', 'physics']
        ),
        load_natural_reasoning(),
        load_nemotron_science_v1(),
        load_medical_o1_verifiable_problem(),
        load_multi_subject_rlvr(),
    ]

    return concatenate_datasets(datasets)
