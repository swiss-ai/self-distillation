from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from data.format import mcq as mcq_format
from data.utils.mcq import coalesce_int, normalize_max_samples
from verl.utils.reward_score.feedback import compute_score as compute_feedback_score
from verl.utils.reward_score.feedback.mcq import flatten_benchmark_metrics

from .base import BenchmarkRunner

if TYPE_CHECKING:
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer


SUPPORTED_MCQ_BENCHMARKS = mcq_format.SUPPORTED_MCQ_BENCHMARKS


@dataclass(frozen=True)
class MCQExample:
    uid: str
    prompt: list[dict[str, str]]
    extra_info: dict[str, Any]
    reward_model: dict[str, Any]
    answer: str = ""


def _load_examples(
    benchmark_name: str, benchmark_config, global_max_samples: int
) -> list[MCQExample]:
    supported_benchmarks = {
        "gpqa_diamond": mcq_format.load_gpqa_diamond_examples,
        "medmcqa": mcq_format.load_medmcqa_examples,
        "medqa_en": mcq_format.load_medqa_examples,
        "mmlu": mcq_format.load_mmlu_examples,
        "mmlu_pro": mcq_format.load_mmlu_pro_examples,
    }
    rows = supported_benchmarks[benchmark_name](benchmark_config, global_max_samples)
    return [
        MCQExample(
            uid=row["uid"],
            prompt=row["prompt"],
            extra_info=dict(row["extra_info"]),
            reward_model=dict(row["reward_model"]),
            answer=row.get("answer", ""),
        )
        for row in rows
    ]


class MCQBenchmarkRunner(BenchmarkRunner[MCQExample]):
    family = "mcq"

    def _evaluate_output(
        self,
        output_text: str,
        batch_examples: list[MCQExample],
        benchmark_non_tensor_batch,
        index: int,
        response_length: int,
        max_response_length: int,
    ) -> dict[str, float | str | None]:
        return compute_feedback_score(
            scoring_module=benchmark_non_tensor_batch["reward_model"][index]["scoring_module"],
            solution_str=output_text,
            ground_truth=benchmark_non_tensor_batch["reward_model"][index]["ground_truth"],
            extra_info={
                "truncated": response_length == max_response_length,
                "reward_model_info": benchmark_non_tensor_batch["reward_model"][index]["info"],
            },
        )

    def _metric_paths(self, benchmark_non_tensor_batch, index: int) -> tuple[str, ...]:
        metric_paths = tuple(benchmark_non_tensor_batch["extra_info"][index].get("metric_paths", ()))
        return metric_paths or super()._metric_paths(benchmark_non_tensor_batch, index)

    def _append_result_metrics(self, infos_dict: dict[str, list], result: dict[str, float | str | None]) -> None:
        infos_dict["acc"].append(result["acc"])
        infos_dict["incorrect_format"].append(result["incorrect_format"])
        infos_dict["truncated"].append(result["truncated"])
        infos_dict["response_length"].append(result["response_length"])
        infos_dict["pred"].append(result["pred"] or "")

    def _flatten_metrics(self, processed_metrics: dict[str, dict[str, dict[str, float]]]) -> dict[str, float]:
        return flatten_benchmark_metrics(processed_metrics)

    def _iter_log_result_fields(self, result: dict[str, float | str | None]) -> list[tuple[str, float | str | None]]:
        return [
            ("pred", result["pred"]),
            ("acc", result["acc"]),
            ("incorrect_format", result["incorrect_format"]),
            ("truncated", result["truncated"]),
            ("response_length", result["response_length"]),
        ]


def build_benchmark_runners(config) -> list[MCQBenchmarkRunner]:
    from .math import SUPPORTED_MATH_BENCHMARKS

    benchmark_config = config.trainer.get("benchmarks", None)
    if benchmark_config is None:
        return []

    enabled = list(benchmark_config.get("enabled", []))
    if not enabled:
        return []

    global_batch_size = int(benchmark_config.get("batch_size", 32))
    global_max_samples = normalize_max_samples(benchmark_config.get("max_samples", -1))
    global_num_generations = benchmark_config.get("num_generations", None)
    global_prompt_length = benchmark_config.get("prompt_length", None)
    mcq_config = benchmark_config.get("mcq", {})
    default_num_generations = coalesce_int(
        mcq_config.get("num_generations", global_num_generations),
        int(config.actor_rollout_ref.rollout.val_kwargs.n),
    )
    default_prompt_length = coalesce_int(
        mcq_config.get("prompt_length", global_prompt_length),
        int(config.actor_rollout_ref.rollout.prompt_length),
    )

    runners: list[MCQBenchmarkRunner] = []
    for benchmark_name in enabled:
        if benchmark_name in SUPPORTED_MATH_BENCHMARKS:
            continue
        if benchmark_name not in SUPPORTED_MCQ_BENCHMARKS:
            raise ValueError(
                f"Unsupported benchmark '{benchmark_name}'. Supported benchmarks: "
                f"{sorted(SUPPORTED_MCQ_BENCHMARKS)}"
            )

        benchmark_specific_config = mcq_config.get(benchmark_name, {})
        examples = _load_examples(benchmark_name, benchmark_specific_config, global_max_samples)
        batch_size = coalesce_int(benchmark_specific_config.get("batch_size", None), global_batch_size)
        num_generations = coalesce_int(
            benchmark_specific_config.get("num_generations", None),
            default_num_generations,
        )
        prompt_length = coalesce_int(
            benchmark_specific_config.get("prompt_length", None),
            default_prompt_length,
        )
        runners.append(
            MCQBenchmarkRunner(
                name=benchmark_name,
                examples=examples,
                batch_size=batch_size,
                num_generations=num_generations,
                prompt_length=prompt_length,
            )
        )

    return runners


__all__ = [
    "MCQExample",
    "MCQBenchmarkRunner",
    "SUPPORTED_MCQ_BENCHMARKS",
    "build_benchmark_runners",
]
