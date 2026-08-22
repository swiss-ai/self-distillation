import json
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from data.prompts.common import (PROMPT_DATASET_COLUMN_NAMES,
                                 PROMPT_DATASET_FEATURES,
                                 make_single_turn_prompt)
from datasets import Dataset, concatenate_datasets, load_dataset

OPEN_CODE_REASONING_2_DATA_SOURCE = "nvidia/OpenCodeReasoning-2"
NEMOTRON_CASCADE_SFT_DATA_SOURCE = "nvidia/Nemotron-Cascade-SFT-Stage-1"
NEMOTRON_COMPETITIVE_PROGRAMMING_DATA_SOURCE = "nvidia/Nemotron-Competitive-Programming-v1"

_NEMOTRON_CP_INFINIBYTE_PARTS = [
    ("infinibyte_part_00", "hf://datasets/nvidia/Nemotron-Competitive-Programming-v1/data/infinibyte.part_00.jsonl"),
    ("infinibyte_part_01", "hf://datasets/nvidia/Nemotron-Competitive-Programming-v1/data/infinibyte.part_01.jsonl"),
]
_NEMOTRON_CP_INFINIBYTE_SAMPLE_SIZE = 50_000
_NEMOTRON_CP_INFINIBYTE_SEED = 42

OCR2_INSTRUCTION_PREFIX = {
    "cpp": (
        "Solve the following programming problem. Explain your approach, "
        "then provide the complete solution in C++.\n\n"
        "Your final code must be in a single ```cpp block:\n"
        "```cpp\n// your code here\n```\n\n"
    ),
    "python": (
        "Solve the following programming problem. Explain your approach, "
        "then provide the complete solution in Python.\n\n"
        "Your final code must be in a single ```python block:\n"
        "```python\n# your code here\n```\n\n"
    ),
}


def _load_ocr2_source_dataset(src_name: str):
    if src_name == "taco":
        return load_dataset(
            "parquet",
            data_files={"train": "hf://datasets/BAAI/TACO/ALL/train-*.parquet"},
            split="train",
        )
    elif src_name == "apps":
        return load_dataset(
            "parquet",
            data_files={
                "train": "hf://datasets/codeparrot/apps@refs/pr/5/all/train-*",
                "test": "hf://datasets/codeparrot/apps@refs/pr/5/all/test-*",
            },
        )
    elif src_name == "code_contests":
        return load_dataset("deepmind/code_contests")
    elif src_name == "open-r1/codeforces":
        return load_dataset("open-r1/codeforces")
    else:
        raise ValueError(f"Unknown source dataset: {src_name}")


def _extract_question_from_source(src_name: str, src_row: Dict[str, Any]) -> str:
    if src_name in ("taco", "apps"):
        return src_row.get("question", "")
    elif src_name == "code_contests":
        return src_row.get("description", "")
    elif src_name == "open-r1/codeforces":
        question = src_row.get("description", "")
        if src_row.get("input_format"):
            question += "\n\nInput\n\n" + src_row["input_format"]
        if src_row.get("output_format"):
            question += "\n\nOutput\n\n" + src_row["output_format"]
        if src_row.get("examples"):
            question += "\n\nExamples"
            for example in src_row["examples"]:
                if "input" in example:
                    question += "\n\nInput\n\n" + example["input"]
                if "output" in example:
                    question += "\n\nOutput\n\n" + example["output"]
        if src_row.get("note"):
            question += "\n\nNote\n\n" + src_row["note"]
        return question
    return ""


def _reconstruct_question(
    example: Dict[str, Any], source_cache: Dict[str, Any]
) -> str:
    src_name = example.get("dataset")
    src_split = example.get("split")
    src_idx = example.get("index")

    if not src_name or not src_split or src_idx is None:
        raise ValueError(
            f"Missing source metadata: dataset={src_name}, split={src_split}, index={src_idx}"
        )

    src_idx = int(src_idx)

    if src_name not in source_cache:
        source_cache[src_name] = _load_ocr2_source_dataset(src_name)

    src_ds = source_cache[src_name]
    if hasattr(src_ds, "keys"):  # DatasetDict
        src_row = src_ds[src_split][src_idx]
    else:  # Dataset
        src_row = src_ds[src_idx]
    question = _extract_question_from_source(src_name, src_row)
    if not question.strip():
        raise ValueError(
            f"Empty question after reconstruction: dataset={src_name}, split={src_split}, index={src_idx}"
        )
    return question


def _map_ocr2_example(
    example: Dict[str, Any],
    idx: int,
    source_cache: Dict[str, Any],
    split: str,
    selection_strategy: str,
) -> Dict[str, Any]:
    question = _reconstruct_question(example, source_cache)
    prompt_text = OCR2_INSTRUCTION_PREFIX[split] + question

    reference = {}
    if example.get("r1_generation"):
        reference["candidate_solution"] = example["r1_generation"]
    if pd.notna(example.get("pass_rate_num")):
        reference["pass_rate"] = float(example["pass_rate_num"])
    if example.get("judgement") is not None:
        reference["judgement"] = str(example["judgement"])

    question_id = example.get("question_id")

    meta_information = {
        "original_dataset": example.get("dataset"),
        "original_split": example.get("split"),
        "original_index": int(example["index"]) if example.get("index") is not None else None,
        "question_id": question_id,
        "difficulty": example.get("difficulty"),
        "source": example.get("source"),
        "selection_strategy": selection_strategy,
        "num_attempts": int(example["num_attempts"]),
        "has_right_attempt": bool(example["has_right_attempt"]),
        "max_pass_rate": float(example["max_pass_rate"]) if pd.notna(example.get("max_pass_rate")) else None,
    }

    return {
        "prompt": make_single_turn_prompt(prompt_text),
        "reference": reference,
        "data_source": OPEN_CODE_REASONING_2_DATA_SOURCE,
        "meta_information": meta_information,
        "data_source_id": str(example["id"]),
        "turn": 0,
    }


def _load_ocr2_split(
    split: str,
    max_rows: Optional[int] = None,
) -> Dataset:
    """
    Load a single OCR2 split as one row per unique task.

    Groups by question_id, reconstructs problem statements from source datasets,
    and selects the best OCR2 attempt per problem (highest pass_rate, prefer
    judgement=="right", longer solution as tie-break).
    """
    selection_strategy = "max_pass_rate_then_prefer_right_then_longer_solution"

    ds = load_dataset(OPEN_CODE_REASONING_2_DATA_SOURCE, split=split)
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))

    df = ds.to_pandas()
    problem_id_col = "question_id" if "question_id" in df.columns else "id"

    df["pass_rate_num"] = pd.to_numeric(df["pass_rate"], errors="coerce")
    df["is_right"] = df["judgement"].astype(str).str.lower().eq("right")
    df["solution_len"] = df["r1_generation"].fillna("").astype(str).str.len()

    problem_stats = df.groupby(problem_id_col, dropna=False).agg(
        num_attempts=("id", "size"),
        has_right_attempt=("is_right", "max"),
        max_pass_rate=("pass_rate_num", "max"),
    ).reset_index()
    df = df.merge(problem_stats, on=problem_id_col, how="left")

    best_per_problem = (
        df.sort_values(
            by=["pass_rate_num", "is_right", "solution_len"],
            ascending=[False, False, False],
            na_position="last",
        )
        .drop_duplicates(subset=[problem_id_col], keep="first")
    )

    best_ds = Dataset.from_pandas(best_per_problem, preserve_index=False)
    source_cache: Dict[str, Any] = {}

    best_ds = best_ds.map(
        lambda example, idx: _map_ocr2_example(example, idx, source_cache, split, selection_strategy),
        with_indices=True,
        remove_columns=best_ds.column_names,
        features=PROMPT_DATASET_FEATURES,
    )
    return best_ds.select_columns(PROMPT_DATASET_COLUMN_NAMES)


def load_open_code_reasoning_2_cpp(max_rows: Optional[int] = None) -> Dataset:
    return _load_ocr2_split("cpp", max_rows=max_rows)


def load_open_code_reasoning_2_python(max_rows: Optional[int] = None) -> Dataset:
    return _load_ocr2_split("python", max_rows=max_rows)


def load_open_code_reasoning_2(max_rows: Optional[int] = None) -> Dataset:
    return concatenate_datasets([
        load_open_code_reasoning_2_cpp(max_rows=max_rows),
        load_open_code_reasoning_2_python(max_rows=max_rows),
    ])


def _extract_first_message(messages, role: str) -> Optional[str]:
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            return None
    if not isinstance(messages, list):
        return None
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == role:
            return msg.get("content")
    return None


def load_nemotron_cascade_sft_code() -> Dataset:
    """
    Load nvidia/Nemotron-Cascade-SFT-Stage-1 (code split), deduplicated
    to one row per unique user prompt.

    Original columns:
        messages: list (user/assistant turns)
        source: str (original data source)
        generator: str (model that produced the response)
    """
    ds = load_dataset(
        NEMOTRON_CASCADE_SFT_DATA_SOURCE, "code", split="train"
    )

    all_messages = ds["messages"]
    seen: set = set()
    unique_indices: List[int] = []
    for idx, messages in enumerate(all_messages):
        problem = _extract_first_message(messages, "user")
        if problem is not None and problem not in seen:
            seen.add(problem)
            unique_indices.append(idx)

    deduped = ds.select(unique_indices)
    original_ids = unique_indices

    def _map_cascade_example(
        example: Dict[str, Any], idx: int
    ) -> Dict[str, Any]:
        problem = _extract_first_message(example["messages"], "user")
        answer = _extract_first_message(example["messages"], "assistant")

        reference: Dict[str, Any] = {}
        if answer is not None:
            reference["expert_solution"] = answer

        return {
            "prompt": make_single_turn_prompt(problem),
            "reference": reference,
            "data_source": NEMOTRON_CASCADE_SFT_DATA_SOURCE,
            "meta_information": {
                "original_data_source": example.get("source"),
                "expert_generator_model": example.get("generator"),
            },
            "data_source_id": str(original_ids[idx]),
            "turn": 0,
        }

    deduped = deduped.map(
        _map_cascade_example,
        with_indices=True,
        remove_columns=deduped.column_names,
        features=PROMPT_DATASET_FEATURES,
    )
    return deduped.select_columns(PROMPT_DATASET_COLUMN_NAMES)


def load_nemotron_cp_infinibyte() -> Dataset:
    """
    Load nvidia/Nemotron-Competitive-Programming-v1 infinibyte splits,
    keep only rows with used_in == ['nano_v3'] (~300k), then randomly
    sample 50k rows (seed=42).
    """
    part_datasets = []
    part_boundaries = [0]
    for _, data_file in _NEMOTRON_CP_INFINIBYTE_PARTS:
        part = load_dataset(
            "json", data_files={"train": data_file}, split="train"
        )
        part_datasets.append(part)
        part_boundaries.append(part_boundaries[-1] + len(part))

    ds = concatenate_datasets(part_datasets)

    ds = ds.add_column("_concat_idx", list(range(len(ds))))
    ds = ds.filter(lambda x: x["used_in"] == ["nano_v3"])
    ds = ds.shuffle(seed=_NEMOTRON_CP_INFINIBYTE_SEED).select(
        range(min(_NEMOTRON_CP_INFINIBYTE_SAMPLE_SIZE, len(ds)))
    )

    def _recover_split(concat_idx: int):
        for i in range(len(_NEMOTRON_CP_INFINIBYTE_PARTS) - 1, -1, -1):
            if concat_idx >= part_boundaries[i]:
                return _NEMOTRON_CP_INFINIBYTE_PARTS[i][0], concat_idx - part_boundaries[i]
        return _NEMOTRON_CP_INFINIBYTE_PARTS[0][0], concat_idx

    def _map_example(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        problem = _extract_first_message(example["messages"], "user")
        answer = _extract_first_message(example["messages"], "assistant")

        reference: Dict[str, Any] = {}
        if answer is not None:
            reference["expert_solution"] = answer

        original_split, original_idx = _recover_split(example["_concat_idx"])

        return {
            "prompt": make_single_turn_prompt(problem),
            "reference": reference,
            "data_source": NEMOTRON_COMPETITIVE_PROGRAMMING_DATA_SOURCE,
            "meta_information": {
                "original_split": original_split,
                "original_index": original_idx,
            },
            "data_source_id": example["uuid"],
            "turn": 0,
        }

    ds = ds.map(
        _map_example,
        with_indices=True,
        remove_columns=ds.column_names,
        features=PROMPT_DATASET_FEATURES,
    )
    return ds.select_columns(PROMPT_DATASET_COLUMN_NAMES)


CODE_DATASETS = [
    OPEN_CODE_REASONING_2_DATA_SOURCE,
    NEMOTRON_CASCADE_SFT_DATA_SOURCE,
    NEMOTRON_COMPETITIVE_PROGRAMMING_DATA_SOURCE,
]

CODE_DATASET_LOADERS: Dict[str, Callable[[], Dataset]] = {
    OPEN_CODE_REASONING_2_DATA_SOURCE: load_open_code_reasoning_2,
    NEMOTRON_CASCADE_SFT_DATA_SOURCE: load_nemotron_cascade_sft_code,
    NEMOTRON_COMPETITIVE_PROGRAMMING_DATA_SOURCE: load_nemotron_cp_infinibyte,
}


def load_code() -> Dataset:
    datasets = []
    for ds_path in CODE_DATASETS:
        loader = CODE_DATASET_LOADERS.get(ds_path)
        if loader is None:
            raise ValueError(f"Unsupported code dataset: {ds_path}")
        datasets.append(loader())
    return concatenate_datasets(datasets)
