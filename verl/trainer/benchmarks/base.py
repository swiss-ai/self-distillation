from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

import numpy as np
import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.metric_utils import process_validation_metrics

if TYPE_CHECKING:
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class BenchmarkExample(Protocol):
    uid: str
    prompt: list[dict[str, str]]
    extra_info: dict[str, Any]
    reward_model: dict[str, Any]
    answer: str


ExampleT = TypeVar("ExampleT", bound=BenchmarkExample)


class BenchmarkRunner(ABC, Generic[ExampleT]):
    family: str

    def __init__(self, name: str, examples: list[ExampleT], batch_size: int, num_generations: int, prompt_length: int):
        self.name = name
        self.examples = examples
        self.batch_size = max(1, batch_size)
        self.num_generations = max(1, num_generations)
        self.prompt_length = int(prompt_length)

    def _build_batch(self, examples: list[ExampleT], start_index: int, global_steps: int) -> DataProto:
        batch = DataProto.from_single_dict(
            {
                "dummy_tensor": torch.zeros((len(examples), 1), dtype=torch.uint8),
                "prompt": np.array([example.prompt for example in examples], dtype=object),
                "raw_prompt": np.array([example.prompt for example in examples], dtype=object),
                "uid": np.array([example.uid for example in examples], dtype=object),
                "answer": np.array([example.answer for example in examples], dtype=object),
                "prompt_length": np.array([self.prompt_length for _ in examples], dtype=object),
                "reward_model": np.array([example.reward_model for example in examples], dtype=object),
                "extra_info": np.array([example.extra_info for example in examples], dtype=object),
                "data_source": np.array(
                    [example.reward_model["scoring_module"] for example in examples],
                    dtype=object,
                ),
                "index": np.arange(start_index, start_index + len(examples)),
                **self._build_extra_batch_fields(examples),
            },
            meta_info={"validate": True, "global_steps": global_steps},
        )
        if self.num_generations > 1:
            batch = batch.repeat(repeat_times=self.num_generations, interleave=True)
            batch.meta_info["validate"] = True
            batch.meta_info["global_steps"] = global_steps
        return batch

    def _build_extra_batch_fields(self, examples: list[ExampleT]) -> dict[str, np.ndarray]:
        return {}

    def _log_first_response(
        self,
        trainer: "RayPPOTrainer",
        example: ExampleT,
        output_text: str,
        result: dict[str, float | str | None],
    ) -> None:
        print(f"benchmark {self.name} first sample begin")
        print(f"[benchmark step] {trainer.global_steps}")
        print("[benchmark prompt]")
        print(example.prompt[-1]["content"])
        print("[benchmark output]")
        print(output_text)
        print("[benchmark ground_truth]")
        print(example.reward_model["ground_truth"])
        for label, value in self._iter_log_result_fields(result):
            print(f"[benchmark {label}]")
            print(value)
        print(f"benchmark {self.name} first sample end")

    def run(self, trainer: "RayPPOTrainer") -> dict[str, float]:
        if not self.examples:
            return {}

        sample_uids: list[str] = []
        data_sources: list[str] = []
        infos_dict: dict[str, list] = defaultdict(list)
        size_divisor = trainer.config.actor_rollout_ref.rollout.agent.num_workers
        logged_first_response = False

        for start_index in range(0, len(self.examples), self.batch_size):
            batch_examples = self.examples[start_index : start_index + self.batch_size]
            benchmark_batch = self._build_batch(
                examples=batch_examples,
                start_index=start_index,
                global_steps=trainer.global_steps,
            )

            benchmark_batch_padded, pad_size = pad_dataproto_to_divisor(benchmark_batch, size_divisor)
            output_batch_padded = trainer.async_rollout_manager.generate_sequences(benchmark_batch_padded)
            output_batch = unpad_dataproto(output_batch_padded, pad_size=pad_size)

            output_ids = output_batch.batch["responses"]
            max_response_length = int(output_ids.shape[-1])
            response_attention_mask = output_batch.batch["attention_mask"][:, -max_response_length:]
            response_lengths = response_attention_mask.sum(dim=-1).cpu().tolist()
            output_texts = [trainer.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            benchmark_non_tensor_batch = benchmark_batch.non_tensor_batch

            for i, output_text in enumerate(output_texts):
                result = self._evaluate_output(
                    output_text=output_text,
                    batch_examples=batch_examples,
                    benchmark_non_tensor_batch=benchmark_non_tensor_batch,
                    index=i,
                    response_length=int(response_lengths[i]),
                    max_response_length=max_response_length,
                )
                result = {
                    **result,
                    "response_length": float(response_lengths[i]),
                    "truncated": float(int(response_lengths[i]) == max_response_length),
                }
                if not logged_first_response:
                    self._log_first_response(
                        trainer=trainer,
                        example=batch_examples[i],
                        output_text=output_text,
                        result=result,
                    )
                    logged_first_response = True
                for metric_path in self._metric_paths(benchmark_non_tensor_batch=benchmark_non_tensor_batch, index=i):
                    sample_uids.append(benchmark_non_tensor_batch["uid"][i])
                    data_sources.append(metric_path)
                    self._append_result_metrics(infos_dict=infos_dict, result=result)

        processed_metrics = process_validation_metrics(
            data_sources=data_sources,
            sample_uids=sample_uids,
            infos_dict=infos_dict,
        )
        return self._flatten_metrics(processed_metrics)

    def _metric_paths(self, benchmark_non_tensor_batch, index: int) -> tuple[str, ...]:
        extra_info = benchmark_non_tensor_batch["extra_info"][index]
        return (str(extra_info.get("data_source", benchmark_non_tensor_batch["data_source"][index])),)

    @abstractmethod
    def _evaluate_output(
        self,
        output_text: str,
        batch_examples: list[ExampleT],
        benchmark_non_tensor_batch,
        index: int,
        response_length: int,
        max_response_length: int,
    ) -> dict[str, float | str | None]:
        raise NotImplementedError

    @abstractmethod
    def _append_result_metrics(self, infos_dict: dict[str, list], result: dict[str, float | str | None]) -> None:
        raise NotImplementedError

    @abstractmethod
    def _flatten_metrics(self, processed_metrics: dict[str, dict[str, dict[str, float]]]) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def _iter_log_result_fields(self, result: dict[str, float | str | None]) -> list[tuple[str, float | str | None]]:
        raise NotImplementedError


__all__ = ["BenchmarkRunner"]
