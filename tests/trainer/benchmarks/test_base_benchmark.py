from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from verl.trainer.benchmarks import base


@dataclass(frozen=True)
class FakeExample:
    uid: str
    prompt: list[dict[str, str]]
    extra_info: dict
    reward_model: dict
    answer: str = ""
    tag: str = ""


class FakeDataProto:
    def __init__(self, non_tensor_batch, meta_info):
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info

    @classmethod
    def from_single_dict(cls, non_tensor_batch, meta_info):
        return cls(non_tensor_batch=non_tensor_batch, meta_info=dict(meta_info))

    def repeat(self, repeat_times, interleave=True):
        repeated_batch = {}
        for key, value in self.non_tensor_batch.items():
            if isinstance(value, np.ndarray):
                repeated_batch[key] = np.repeat(value, repeat_times)
            else:
                repeated_batch[key] = value
        return type(self)(non_tensor_batch=repeated_batch, meta_info=dict(self.meta_info))


class FakeRunner(base.BenchmarkRunner[FakeExample]):
    family = "fake"

    def _build_extra_batch_fields(self, examples: list[FakeExample]) -> dict[str, np.ndarray]:
        return {
            "metric_paths": np.array([tuple(example.extra_info["metric_paths"]) for example in examples], dtype=object),
            "tag": np.array([example.tag for example in examples], dtype=object),
        }

    def _evaluate_output(
        self,
        output_text: str,
        batch_examples: list[FakeExample],
        benchmark_non_tensor_batch,
        index: int,
        response_length: int,
        max_response_length: int,
    ) -> dict[str, float | str | None]:
        return {
            "acc": float(output_text == benchmark_non_tensor_batch["reward_model"][index]["ground_truth"]),
            "pred": output_text,
            "tag": benchmark_non_tensor_batch["tag"][index],
        }

    def _metric_paths(self, benchmark_non_tensor_batch, index: int) -> tuple[str, ...]:
        return tuple(benchmark_non_tensor_batch["metric_paths"][index]) or super()._metric_paths(
            benchmark_non_tensor_batch, index
        )

    def _append_result_metrics(self, infos_dict: dict[str, list], result: dict[str, float | str | None]) -> None:
        infos_dict["acc"].append(result["acc"])
        infos_dict["pred"].append(result["pred"])
        infos_dict["tag"].append(result["tag"])

    def _flatten_metrics(self, processed_metrics: dict[str, dict[str, dict[str, float]]]) -> dict[str, float]:
        return {"flattened_count": processed_metrics["count"]}

    def _iter_log_result_fields(self, result: dict[str, float | str | None]) -> list[tuple[str, float | str | None]]:
        return [("pred", result["pred"]), ("acc", result["acc"])]


def test_benchmark_runner_run_uses_shared_rollout_loop(monkeypatch):
    captured = {}

    monkeypatch.setattr(base, "DataProto", FakeDataProto)
    monkeypatch.setattr(base, "pad_dataproto_to_divisor", lambda batch, size_divisor: (batch, 0))
    monkeypatch.setattr(base, "unpad_dataproto", lambda batch, pad_size: batch)

    def fake_process_validation_metrics(data_sources, sample_uids, infos_dict):
        captured["data_sources"] = list(data_sources)
        captured["sample_uids"] = list(sample_uids)
        captured["infos_dict"] = {key: list(value) for key, value in infos_dict.items()}
        return {"count": len(data_sources)}

    monkeypatch.setattr(base, "process_validation_metrics", fake_process_validation_metrics)

    trainer = SimpleNamespace(
        global_steps=11,
        config=SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(agent=SimpleNamespace(num_workers=1))
            )
        ),
        tokenizer=SimpleNamespace(
            decode=lambda ids, skip_special_tokens=True: {65: "A", 66: "B"}[int(ids[0])]
        ),
        async_rollout_manager=SimpleNamespace(
            generate_sequences=lambda _batch: SimpleNamespace(
                batch={
                    "responses": torch.tensor([[65], [66]], dtype=torch.long),
                    "attention_mask": torch.tensor([[1], [1]], dtype=torch.long),
                }
            )
        ),
    )

    runner = FakeRunner(
        name="fake-benchmark",
        examples=[
            FakeExample(
                uid="example-1",
                prompt=[{"role": "user", "content": "Prompt 1"}],
                extra_info={"data_source": "fake/source", "metric_paths": ("fake/source/alias-a", "fake/source/alias-b")},
                reward_model={"ground_truth": "A", "scoring_module": "fake", "info": {}},
                tag="first",
            ),
            FakeExample(
                uid="example-2",
                prompt=[{"role": "user", "content": "Prompt 2"}],
                extra_info={"data_source": "fake/source", "metric_paths": ()},
                reward_model={"ground_truth": "B", "scoring_module": "fake", "info": {}},
                tag="second",
            ),
        ],
        batch_size=2,
        num_generations=1,
        prompt_length=128,
    )

    metrics = runner.run(trainer)

    assert metrics == {"flattened_count": 3}
    assert captured["sample_uids"] == ["example-1", "example-1", "example-2"]
    assert captured["data_sources"] == ["fake/source/alias-a", "fake/source/alias-b", "fake/source"]
    assert captured["infos_dict"]["acc"] == [1.0, 1.0, 1.0]
    assert captured["infos_dict"]["pred"] == ["A", "A", "B"]
    assert captured["infos_dict"]["tag"] == ["first", "first", "second"]
