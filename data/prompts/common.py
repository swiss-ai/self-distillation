from collections import defaultdict
from typing import Any, Dict, List, TypedDict

from datasets.features import Json

from datasets import Dataset, Features, Value


class PromptDatasetColumn(TypedDict):
    name: str
    dtype: str
    description: str


PROMPT_DATASET_FEATURES = Features({
    "prompt": [
        {
            "role": Value("string"),
            "content": Value("string"),
        }
    ],
    "reference": Json(),
    "data_source": Value("string"),
    "meta_information": Json(),
    "data_source_id": Value("string"),
    "turn": Value("int64"),
})

PROMPT_DATASET_COLUMN_NAMES = list(PROMPT_DATASET_FEATURES)


def make_single_turn_prompt(user_content: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": user_content}]


def normalize_message(message: Dict[str, Any]) -> Dict[str, str]:
    return {
        "role": str(message["role"]),
        "content": str(message["content"]),
    }

def subsample_dataset(
    dataset: Dataset, 
    num_samples_per_source: int,
    seed: int = 42,
) -> Dataset:
    shuffled_dataset = dataset.shuffle(seed=seed)
    selected_counts = defaultdict(int)
    selected_indices = []

    for idx, data_source in enumerate(shuffled_dataset["data_source"]):
        if selected_counts[data_source] >= num_samples_per_source:
            continue

        selected_indices.append(idx)
        selected_counts[data_source] += 1

    return shuffled_dataset.select(selected_indices)
