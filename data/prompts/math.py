import json
from typing import Any, Callable, Dict

from data.prompts.common import (PROMPT_DATASET_COLUMN_NAMES,
                                 PROMPT_DATASET_FEATURES,
                                 make_single_turn_prompt)
from datasets import Dataset, concatenate_datasets, load_dataset

NEMOTRON_MATH_V2_DATA_SOURCE = "nvidia/Nemotron-Math-v2"
DEEPMATH_103K_DATA_SOURCE = "zwhe99/DeepMath-103K"
NUMINA_MATH_RL_DATA_SOURCE = "nlile/NuminaMath-1.5-RL-Verifiable"
DAPO_MATH_17K_DATA_SOURCE = "open-r1/DAPO-Math-17k-Processed"
BIG_MATH_RL_DATA_SOURCE = "open-r1/Big-Math-RL-Verified-Processed"

MATH_INSTRUCTION_PREFIX = (
    "Solve the following math problem. Make sure to put the answer "
    "(and only the answer) inside \\boxed{}.\n\n"
)

MATH_DATASETS = [
    NEMOTRON_MATH_V2_DATA_SOURCE,
    DEEPMATH_103K_DATA_SOURCE,
    NUMINA_MATH_RL_DATA_SOURCE,
    DAPO_MATH_17K_DATA_SOURCE,
    BIG_MATH_RL_DATA_SOURCE,
]

SPLIT_PRIORITY = {
    "high_part00": 0, "high_part01": 0, "high_part02": 0,
    "medium": 1,
    "low": 2,
}


def _dedupe_by_problem(ds_dict) -> Dataset:
    """Pick one row per unique problem with priority: high > medium > low."""
    seen = {}
    indices_per_split = {s: [] for s in ds_dict}

    for split in sorted(ds_dict.keys(), key=lambda s: SPLIT_PRIORITY.get(s, 99)):
        problems = ds_dict[split]["problem"]
        for i, p in enumerate(problems):
            if p not in seen:
                seen[p] = split
                indices_per_split[split].append(i)

    subsets = []
    for split, idxs in indices_per_split.items():
        if idxs:
            subset = ds_dict[split].select(idxs)
            subset = subset.add_column("_source_split", [split] * len(subset))
            subset = subset.add_column("_original_idx", idxs)
            subsets.append(subset)

    return concatenate_datasets(subsets)


def load_nemotron_math_v2() -> Dataset:
    """
    Load nvidia/Nemotron-Math-v2 dataset, deduplicated to one row per unique problem.

    ~324K unique problems from ~7M reasoning trajectories.
    Priority: high reasoning splits first, then medium, then low.

    Original columns:
        problem: str
        messages: list (user/assistant turns)
        expected_answer: str
        changed_answer_to_majority: bool
        metadata: str (JSON with pass rates per reasoning config)
        data_source: str (AoPS or StackExchange-Math)
        tool: str
        url: str
        user_url: str
        user_name: str
    """

    def map_nemotron_math_example(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        meta_raw = example.get("metadata")
        if isinstance(meta_raw, str):
            meta_raw = json.loads(meta_raw)

        source_split = example["_source_split"]
        original_idx = example["_original_idx"]

        messages = example.get("messages", [])
        expert_solution = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                expert_solution = msg["content"]
                break

        reference = {"expected_answer": example["expected_answer"]}
        if expert_solution is not None:
            reference["expert_solution"] = expert_solution

        return {
            "prompt": make_single_turn_prompt(
                MATH_INSTRUCTION_PREFIX + example["problem"]
            ),
            "reference": reference,
            "data_source": NEMOTRON_MATH_V2_DATA_SOURCE,
            "meta_information": {
                "changed_answer_to_majority": example.get("changed_answer_to_majority"),
                "original_data_source": example.get("data_source"),
                "source_split": source_split,
                "difficulty": meta_raw,
                "tool": example.get("tool"),
                "url": example.get("url"),
            },
            "data_source_id": f"{source_split}/{original_idx}",
            "turn": 0,
        }

    ds_dict = load_dataset(NEMOTRON_MATH_V2_DATA_SOURCE)
    deduped = _dedupe_by_problem(ds_dict)

    deduped = deduped.map(
        map_nemotron_math_example,
        with_indices=True,
        remove_columns=deduped.column_names,
        features=PROMPT_DATASET_FEATURES,
    )
    return deduped.select_columns(PROMPT_DATASET_COLUMN_NAMES)


def load_deepmath_103k() -> Dataset:
    """
    Load DeepMath-103K dataset — challenging, decontaminated math problems.

    Rows: ~103K

    Original columns:
        question: str
        final_answer: str
        difficulty: float64
        topic: str
        r1_solution_1: str (DeepSeek-R1 reasoning path)
        r1_solution_2: str
        r1_solution_3: str
    """

    def map_deepmath_example(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        reference: Dict[str, Any] = {"expected_answer": example["final_answer"]}
        if example.get("r1_solution_1"):
            reference["expert_solution"] = example["r1_solution_1"]

        return {
            "prompt": make_single_turn_prompt(
                MATH_INSTRUCTION_PREFIX + example["question"]
            ),
            "reference": reference,
            "data_source": DEEPMATH_103K_DATA_SOURCE,
            "meta_information": {
                "difficulty": example.get("difficulty"),
                "topic": example.get("topic"),
            },
            "data_source_id": str(idx),
            "turn": 0,
        }

    ds = load_dataset(DEEPMATH_103K_DATA_SOURCE, split="train")
    ds = ds.map(
        map_deepmath_example,
        with_indices=True,
        remove_columns=ds.column_names,
        features=PROMPT_DATASET_FEATURES,
    )
    return ds.select_columns(PROMPT_DATASET_COLUMN_NAMES)


def load_numina_math_rl() -> Dataset:
    """
    Load NuminaMath-1.5-RL-Verifiable — curated math word problems with
    verifiable numerical answers from olympiads, contests, and forums.

    Rows: ~131K

    Original columns:
        problem: str
        solution: str (step-by-step reference solution)
        answer: str (definitive numerical answer)
        problem_type: str (Algebra, Geometry, Number Theory, etc.)
        question_type: str
        source: str (olympiads, cn_contest, aops_forum, etc.)
        problem_is_valid: str
        solution_is_valid: str
        synthetic: bool
    """

    def map_numina_example(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        reference: Dict[str, Any] = {"expected_answer": example["answer"]}
        if example.get("solution"):
            reference["expert_solution"] = example["solution"]

        return {
            "prompt": make_single_turn_prompt(
                MATH_INSTRUCTION_PREFIX + example["problem"]
            ),
            "reference": reference,
            "data_source": NUMINA_MATH_RL_DATA_SOURCE,
            "meta_information": {
                "problem_type": example.get("problem_type"),
                "question_type": example.get("question_type"),
                "source": example.get("source"),
            },
            "data_source_id": str(idx),
            "turn": 0,
        }

    ds = load_dataset(NUMINA_MATH_RL_DATA_SOURCE, split="train")
    ds = ds.map(
        map_numina_example,
        with_indices=True,
        remove_columns=ds.column_names,
        features=PROMPT_DATASET_FEATURES,
    )
    return ds.select_columns(PROMPT_DATASET_COLUMN_NAMES)


def load_dapo_math_17k() -> Dataset:
    """
    Load open-r1/DAPO-Math-17k-Processed (en split) — curated math problems
    from the DAPO project with step-by-step prompts and ground-truth answers.

    Rows: ~17K

    Original columns:
        prompt: str (math problem text)
        solution: str (ground-truth answer)
        data_source: str
        source_prompt: list[dict] (messages with role/content)
        ability: str
        reward_model: dict (ground_truth, style)
        extra_info: dict (index)
    """

    def map_dapo_example(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        messages = example.get("source_prompt", [])
        content = ""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg["content"]
                break

        return {
            "prompt": make_single_turn_prompt(content),
            "reference": {"expected_answer": example["solution"]},
            "data_source": DAPO_MATH_17K_DATA_SOURCE,
            "meta_information": {"split": "en"},
            "data_source_id": str(idx),
            "turn": 0,
        }

    ds = load_dataset(DAPO_MATH_17K_DATA_SOURCE, "en", split="train")
    ds = ds.map(
        map_dapo_example,
        with_indices=True,
        remove_columns=ds.column_names,
        features=PROMPT_DATASET_FEATURES,
    )
    return ds.select_columns(PROMPT_DATASET_COLUMN_NAMES)


def load_big_math_rl() -> Dataset:
    """
    Load open-r1/Big-Math-RL-Verified-Processed (all config) — large-scale
    RL-verified math problems with solve-rate metadata.

    Original columns:
        prompt: str (math problem text)
        solution: str (ground-truth answer)
        source: str (original data source)
        domain: list[str] (math domain tags)
        llama8b_solve_rate: float
    """

    def map_big_math_rl_example(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        return {
            "prompt": make_single_turn_prompt(example["prompt"]),
            "reference": {"expected_answer": example["solution"]},
            "data_source": BIG_MATH_RL_DATA_SOURCE,
            "meta_information": {
                "original_data_source": example.get("source"),
                "domain": example.get("domain"),
                "llama8b_solve_rate": example.get("llama8b_solve_rate"),
            },
            "data_source_id": str(idx),
            "turn": 0,
        }

    ds = load_dataset(BIG_MATH_RL_DATA_SOURCE, "all", split="train")
    ds = ds.map(
        map_big_math_rl_example,
        with_indices=True,
        remove_columns=ds.column_names,
        features=PROMPT_DATASET_FEATURES,
    )
    return ds.select_columns(PROMPT_DATASET_COLUMN_NAMES)


MATH_DATASET_LOADERS: Dict[str, Callable[[], Dataset]] = {
    NEMOTRON_MATH_V2_DATA_SOURCE: load_nemotron_math_v2,
    DEEPMATH_103K_DATA_SOURCE: load_deepmath_103k,
    NUMINA_MATH_RL_DATA_SOURCE: load_numina_math_rl,
    DAPO_MATH_17K_DATA_SOURCE: load_dapo_math_17k,
    BIG_MATH_RL_DATA_SOURCE: load_big_math_rl,
}


def load_math() -> Dataset:
    datasets = []
    for ds_path in MATH_DATASETS:
        loader = MATH_DATASET_LOADERS.get(ds_path)
        if loader is None:
            raise ValueError(f"Unsupported math dataset: {ds_path}")
        datasets.append(loader())

    return concatenate_datasets(datasets)
