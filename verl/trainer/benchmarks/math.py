from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from data.format.math import load_math
from verl.utils.reward_score.feedback import compute_score as compute_feedback_score

from .base import BenchmarkRunner

if TYPE_CHECKING:
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer


BENCHMARK_TO_DATASET = {
    "math500": "math-ai/math500",
    "gsm8k": "openai/gsm8k",
    "aime24": "math-ai/aime24",
    "aime25": "math-ai/aime25",
    "amc23": "math-ai/amc23",
}

SUPPORTED_MATH_BENCHMARKS = frozenset(BENCHMARK_TO_DATASET.keys())


@dataclass(frozen=True)
class MathExample:
    uid: str
    prompt: list[dict[str, str]]
    extra_info: dict[str, Any]
    reward_model: dict[str, Any]
    answer: str = ""


def _normalize_max_samples(value) -> int:
    if value is None:
        return -1
    return int(value)


def _coalesce_int(value, fallback: int) -> int:
    if value is None:
        return fallback
    return int(value)


def evaluate_math_response(
    solution: str,
    ground_truth: str,
    *,
    scoring_module: str = "math",
    reward_model_info: dict[str, Any] | None = None,
    response_length: int = 0,
    max_response_length: int = -1,
) -> dict[str, float | str]:
    result = compute_feedback_score(
        scoring_module=scoring_module,
        solution_str=solution,
        ground_truth=ground_truth,
        extra_info={
            "truncated": response_length == max_response_length,
            "reward_model_info": reward_model_info or {},
        },
    )
    result["no_answer"] = float(result["incorrect_format"])
    return result


def _load_examples(
    benchmark_name: str, benchmark_config, global_max_samples: int
) -> list[MathExample]:
    dataset_name = benchmark_config.get(
        "dataset_name", BENCHMARK_TO_DATASET[benchmark_name]
    )
    ds = load_math(dataset_name)

    max_samples = _normalize_max_samples(
        benchmark_config.get("max_samples", global_max_samples)
    )
    if max_samples == 0:
        return []
    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    examples = []
    for i, row in enumerate(ds):
        examples.append(
            MathExample(
                uid=f"{benchmark_name}:{i}",
                prompt=[{"role": "user", "content": row["prompt"]}],
                extra_info={"data_source": f"math/{benchmark_name}"},
                reward_model={
                    "ground_truth": row["answer"],
                    "scoring_module": "math",
                    "info": {},
                },
            )
        )
    return examples


def flatten_math_benchmark_metrics(
    processed_metrics: dict[str, dict[str, dict[str, float]]],
) -> dict[str, float]:
    metric_dict: dict[str, float] = {}

    for data_source, var2metric2val in processed_metrics.items():
        for var_name in ("acc", "no_answer", "incorrect_format", "truncated", "response_length"):
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

            for summary_var in ("no_answer", "incorrect_format", "truncated", "response_length"):
                summary_metrics = var2metric2val.get(summary_var, {})
                if mean_key in summary_metrics:
                    metric_dict[f"benchmark/{data_source}/{summary_var}"] = summary_metrics[mean_key]

    return metric_dict


class MathBenchmarkRunner(BenchmarkRunner[MathExample]):
    family = "math"

    def _evaluate_output(
        self,
        output_text: str,
        batch_examples: list[MathExample],
        benchmark_non_tensor_batch,
        index: int,
        response_length: int,
        max_response_length: int,
    ) -> dict[str, float | str | None]:
        return evaluate_math_response(
            solution=output_text,
            ground_truth=benchmark_non_tensor_batch["reward_model"][index]["ground_truth"],
            scoring_module=benchmark_non_tensor_batch["reward_model"][index]["scoring_module"],
            reward_model_info=benchmark_non_tensor_batch["reward_model"][index]["info"],
            response_length=response_length,
            max_response_length=max_response_length,
        )

    def _append_result_metrics(self, infos_dict: dict[str, list], result: dict[str, float | str | None]) -> None:
        infos_dict["acc"].append(result["acc"])
        infos_dict["no_answer"].append(result["no_answer"])
        infos_dict["incorrect_format"].append(result["incorrect_format"])
        infos_dict["truncated"].append(result["truncated"])
        infos_dict["response_length"].append(result["response_length"])
        infos_dict["pred"].append(result["pred"])

    def _flatten_metrics(self, processed_metrics: dict[str, dict[str, dict[str, float]]]) -> dict[str, float]:
        return flatten_math_benchmark_metrics(processed_metrics)

    def _iter_log_result_fields(self, result: dict[str, float | str | None]) -> list[tuple[str, float | str | None]]:
        return [
            ("pred", result["pred"]),
            ("acc", result["acc"]),
            ("no_answer", result["no_answer"]),
            ("incorrect_format", result["incorrect_format"]),
            ("truncated", result["truncated"]),
            ("response_length", result["response_length"]),
        ]


def build_benchmark_runners(config) -> list[MathBenchmarkRunner]:
    benchmark_config = config.trainer.get("benchmarks", None)
    if benchmark_config is None:
        return []

    enabled = list(benchmark_config.get("enabled", []))
    if not enabled:
        return []

    global_batch_size = int(benchmark_config.get("batch_size", 32))
    global_max_samples = _normalize_max_samples(benchmark_config.get("max_samples", -1))
    math_config = benchmark_config.get("math", {})
    default_num_generations = _coalesce_int(
        math_config.get("num_generations", None),
        int(config.actor_rollout_ref.rollout.val_kwargs.n),
    )

    runners: list[MathBenchmarkRunner] = []
    for benchmark_name in enabled:
        if benchmark_name not in SUPPORTED_MATH_BENCHMARKS:
            continue

        per_benchmark_config = math_config.get(benchmark_name, {})
        examples = _load_examples(benchmark_name, per_benchmark_config, global_max_samples)

        batch_size = _coalesce_int(per_benchmark_config.get("batch_size", None), global_batch_size)
        num_generations = _coalesce_int(per_benchmark_config.get("num_generations", None), default_num_generations)
        prompt_length = _coalesce_int(
            per_benchmark_config.get("prompt_length", None),
            int(config.actor_rollout_ref.rollout.prompt_length),
        )
        runners.append(
            MathBenchmarkRunner(
                name=benchmark_name,
                examples=examples,
                batch_size=batch_size,
                num_generations=num_generations,
                prompt_length=prompt_length,
            )
        )

    return runners


__all__ = [
    "BENCHMARK_TO_DATASET",
    "SUPPORTED_MATH_BENCHMARKS",
    "MathBenchmarkRunner",
    "build_benchmark_runners",
]
