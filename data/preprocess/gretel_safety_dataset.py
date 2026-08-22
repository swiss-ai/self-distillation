#!/usr/bin/env python3
"""
Convert gretelai/gretel-safety-alignment-en-v1 into the canonical parquet schema.

This is a safety-alignment dataset: each row pairs an adversarial `prompt` with an
unsafe `response` and an aligned `safe_response`, plus LLM-judge scores and
probability-of-harm estimates for both. Rows are labelled with a `risk_category`,
`sub_category`, `tactic`, and `persona`.

The aligned `safe_response` is used as the target answer (override with
--answer-field response to train/evaluate against the unsafe completion instead),
and `risk_category: sub_category` is stored as the reward-model ground truth.

The dataset ships native train/test/validation splits; they are concatenated and
then re-split into a fresh train/test set (controlled by --test-size / --seed),
matching the other scripts in this directory.

Example Command:
    python data/preprocess/gretel_safety_dataset.py \
        --output-dir datasets/gretel-safety-alignment
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the Gretel safety-alignment dataset into canonical parquet files.")
    parser.add_argument("--source", default="gretelai/gretel-safety-alignment-en-v1", help="HF dataset ID or local path.")
    parser.add_argument("--output-dir", required=True, help="Directory where parquet files will be written.")
    parser.add_argument(
        "--subset",
        default=None,
        help="Optional config name (e.g. Discrimination). Defaults to the full 'default' config.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test", "validation"],
        help="Native splits to load and concatenate before re-splitting.",
    )
    parser.add_argument(
        "--answer-field",
        default="safe_response",
        choices=["safe_response", "response"],
        help="Which column to store as the target `answer`.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="Optional Hugging Face token for private or gated datasets.",
    )
    parser.add_argument(
        "--default-data-source",
        default=None,
        help="data_source recorded for every row. Defaults to --source.",
    )
    parser.add_argument(
        "--scoring-module",
        default="llm",
        help="reward_model.scoring_module to assign to every row (e.g. llm, zero).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Fraction of rows to place in the test split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test splitting.")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def convert_split(
    dataset: Dataset,
    *,
    default_data_source: str,
    answer_field: str,
    scoring_module: str,
) -> tuple[list[dict[str, Any]], Counter]:
    rows: list[dict[str, Any]] = []
    filtered: Counter = Counter()

    for row in dataset:
        prompt_text = normalize_text(row.get("prompt"))
        if not prompt_text:
            filtered["empty_prompt"] += 1
            continue

        answer = normalize_text(row.get(answer_field))
        if not answer:
            filtered["empty_answer"] += 1
            continue

        risk_category = normalize_text(row.get("risk_category"))
        sub_category = normalize_text(row.get("sub_category"))

        rows.append(
            {
                "prompt": [{"role": "user", "content": prompt_text}],
                "extra_info": {
                    "data_source": default_data_source,
                    "prompt": prompt_text,
                    "id": row.get("id"),
                    "persona": normalize_text(row.get("persona")),
                    "risk_category": risk_category,
                    "sub_category": sub_category,
                    "tactic": normalize_text(row.get("tactic")),
                    "judge_response_score": row.get("judge_response_score"),
                    "judge_safe_response_score": row.get("judge_safe_response_score"),
                    "response_probability_of_harm": row.get("response_probability_of_harm"),
                    "safe_response_probability_of_harm": row.get("safe_response_probability_of_harm"),
                },
                "reward_model": {
                    "ground_truth": f"{risk_category}: {sub_category}",
                    "scoring_module": scoring_module,
                },
                "answer": answer,
            }
        )

    return rows, filtered


def save_rows_as_parquet(rows: list[dict[str, Any]], output_path: Path) -> None:
    Dataset.from_list(rows).to_parquet(str(output_path))


def main() -> None:
    args = parse_args()
    output_dir = Path(os.path.expanduser(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    default_data_source = args.default_data_source or args.source

    parts = [load_dataset(args.source, args.subset, split=split, token=args.hf_token) for split in args.splits]
    ds = concatenate_datasets(parts)
    print(f"Concatenated {args.splits} into {len(ds)} rows")

    rows, filtered = convert_split(
        ds,
        default_data_source=default_data_source,
        answer_field=args.answer_field,
        scoring_module=args.scoring_module,
    )
    if not rows:
        raise ValueError(f"No valid rows produced. Filter summary: {dict(filtered)}")

    print(f"kept {len(rows)} rows, filtered {dict(filtered)}")

    split_dataset = Dataset.from_list(rows).train_test_split(test_size=args.test_size, seed=args.seed)
    save_rows_as_parquet(list(split_dataset["train"]), output_dir / "train.parquet")
    save_rows_as_parquet(list(split_dataset["test"]), output_dir / "test.parquet")
    print(f"Wrote {output_dir / 'train.parquet'} ({len(split_dataset['train'])} rows)")
    print(f"Wrote {output_dir / 'test.parquet'} ({len(split_dataset['test'])} rows)")


if __name__ == "__main__":
    main()
