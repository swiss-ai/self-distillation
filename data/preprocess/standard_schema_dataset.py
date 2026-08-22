#!/usr/bin/env python3
"""
Convert a dataset in the post-training **standard schema** into the canonical
parquet schema used by the training/eval pipeline.

The standard schema is what the `posttraining-data` pipeline emits (see its
02-standardisation stage), e.g. the decontaminated Gretel dataset. Every row
looks like:

    {
      "conversation_id": str,
      "dataset_source":  str,
      "original_metadata": {...},
      "system_prompt":   {"content": str, "metadata": {...}},
      "initial_prompt":  {"role": "user", "content": str, "metadata": {...}},
      "conversation_branches": [
          {"messages": [{"role": "assistant", "parts": [{"type": "response", "content": str, ...}, ...]}, ...]},
          ...   # >1 branch for preference data (branch 0 = chosen, 1 = rejected)
      ],
      ...
    }

Unlike the per-dataset scripts in this directory, this one is generic: it reads
the nested standard-schema fields rather than flat raw columns, and loads the
dataset with `load_from_disk` (the standard schema is always written via
`save_to_disk`, which `load_dataset` cannot read correctly).

Mapping to the canonical schema (one row -> one training example):
  - prompt      <- system_prompt (optional) + initial_prompt
  - answer      <- the `response` parts of the first assistant message in the
                   selected conversation branch (--branch-index; 0 = chosen)
  - extra_info  <- data_source, conversation_id, prompt text, + original_metadata
  - reward_model.ground_truth <- --ground-truth-template formatted against
                   original_metadata (empty by default)

This is a single-turn extraction: the initial user prompt is the prompt and the
first assistant turn of the chosen branch is the answer. Extra turns inside a
branch are ignored.

Example (decontaminated Gretel, chosen = safe_response):
    python data/preprocess/standard_schema_dataset.py \
        --source /iopsstor/scratch/cscs/smarian/datasets/apertus/gretel/gretel-safety-alignment-en-v1-decontaminated \
        --output-dir datasets/gretel-safety-alignment-decontaminated \
        --ground-truth-template "{risk_category}: {sub_category}"

    # train against the unsafe completion instead (rejected branch):
    #   --branch-index 1
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a standard-schema dataset into canonical parquet files."
    )
    parser.add_argument("--source", required=True, help="Path to a standard-schema dataset (load_from_disk).")
    parser.add_argument("--output-dir", required=True, help="Directory where parquet files will be written.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test", "validation"],
        help="Splits to use (intersected with what's available). Ignored for a single Dataset.",
    )
    parser.add_argument(
        "--branch-index",
        type=int,
        default=0,
        help="Which conversation branch supplies the answer (0 = chosen / primary, 1 = rejected, ...).",
    )
    parser.add_argument(
        "--include-thoughts",
        action="store_true",
        help="Also include `thought` parts (reasoning) in the answer, in their original order.",
    )
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Drop the system prompt from the output prompt even when present.",
    )
    parser.add_argument(
        "--ground-truth-template",
        default="",
        help='reward_model.ground_truth, formatted against original_metadata, e.g. "{risk_category}: {sub_category}".',
    )
    parser.add_argument(
        "--scoring-module",
        default="llm",
        help="reward_model.scoring_module to assign to every row (e.g. llm, zero).",
    )
    parser.add_argument(
        "--default-data-source",
        default=None,
        help="Fallback data_source when a row has no dataset_source. Defaults to --source.",
    )
    parser.add_argument(
        "--keep-native-splits",
        action="store_true",
        help="Write one parquet per native split instead of concatenating and re-splitting.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Fraction of rows in the test split when re-splitting (ignored with --keep-native-splits).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test splitting.")
    return parser.parse_args()


class _SafeDict(dict):
    """dict that renders missing format keys as empty strings."""

    def __missing__(self, key: str) -> str:  # noqa: D401
        return ""


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_prompt(row: dict[str, Any], *, include_system: bool) -> list[dict[str, str]] | None:
    """Build the canonical chat-message prompt list, or None if unusable."""
    messages: list[dict[str, str]] = []

    if include_system:
        system_prompt = row.get("system_prompt") or {}
        system_content = normalize_text(system_prompt.get("content"))
        if system_content:
            messages.append({"role": "system", "content": system_content})

    initial = row.get("initial_prompt") or {}
    content = normalize_text(initial.get("content"))
    if not content:
        return None
    role = normalize_text(initial.get("role")) or "user"
    messages.append({"role": role, "content": content})

    return messages


def extract_answer(row: dict[str, Any], *, branch_index: int, include_thoughts: bool) -> str:
    """Concatenate the response (and optionally thought) parts of the chosen branch's first assistant turn."""
    branches = row.get("conversation_branches") or []
    if branch_index >= len(branches):
        return ""

    messages = branches[branch_index].get("messages") or []
    if not messages:
        return ""

    first = messages[0]
    if normalize_text(first.get("role")) != "assistant":
        return ""

    wanted = {"response", "thought"} if include_thoughts else {"response"}
    pieces = [
        normalize_text(part.get("content"))
        for part in (first.get("parts") or [])
        if part.get("type") in wanted
    ]
    return "\n\n".join(piece for piece in pieces if piece)


def convert_split(
    dataset: Dataset,
    *,
    default_data_source: str,
    branch_index: int,
    include_thoughts: bool,
    include_system: bool,
    ground_truth_template: str,
    scoring_module: str,
) -> tuple[list[dict[str, Any]], Counter]:
    rows: list[dict[str, Any]] = []
    filtered: Counter = Counter()

    for row in dataset:
        prompt = build_prompt(row, include_system=include_system)
        if prompt is None:
            filtered["empty_prompt"] += 1
            continue

        answer = extract_answer(row, branch_index=branch_index, include_thoughts=include_thoughts)
        if not answer:
            filtered["empty_answer"] += 1
            continue

        metadata = row.get("original_metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        prompt_text = prompt[-1]["content"]
        data_source = normalize_text(row.get("dataset_source")) or default_data_source

        ground_truth = ground_truth_template.format_map(
            _SafeDict({key: normalize_text(value) for key, value in metadata.items()})
        )

        extra_info: dict[str, Any] = dict(metadata)
        extra_info.update(
            {
                "data_source": data_source,
                "conversation_id": row.get("conversation_id"),
                "prompt": prompt_text,
            }
        )

        rows.append(
            {
                "prompt": prompt,
                "extra_info": extra_info,
                "reward_model": {
                    "ground_truth": ground_truth,
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

    loaded = load_from_disk(os.path.expanduser(args.source))
    if isinstance(loaded, DatasetDict):
        available = list(loaded.keys())
        selected = [split for split in args.splits if split in available] or available
        print(f"Available splits: {available} -> using {selected}")
        split_to_ds = {split: loaded[split] for split in selected}
    else:
        split_to_ds = {"train": loaded}

    convert_kwargs = dict(
        default_data_source=default_data_source,
        branch_index=args.branch_index,
        include_thoughts=args.include_thoughts,
        include_system=not args.no_system_prompt,
        ground_truth_template=args.ground_truth_template,
        scoring_module=args.scoring_module,
    )

    if args.keep_native_splits:
        for split, ds in split_to_ds.items():
            rows, filtered = convert_split(ds, **convert_kwargs)
            if not rows:
                raise ValueError(f"No valid rows produced for split '{split}'. Filter summary: {dict(filtered)}")
            print(f"[{split}] kept {len(rows)} rows, filtered {dict(filtered)}")
            save_rows_as_parquet(rows, output_dir / f"{split}.parquet")
            print(f"Wrote {output_dir / f'{split}.parquet'} ({len(rows)} rows)")
        return

    combined = concatenate_datasets(list(split_to_ds.values())) if len(split_to_ds) > 1 else next(iter(split_to_ds.values()))
    print(f"Concatenated {list(split_to_ds.keys())} into {len(combined)} rows")

    rows, filtered = convert_split(combined, **convert_kwargs)
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
