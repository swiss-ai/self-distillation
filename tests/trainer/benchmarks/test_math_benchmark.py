from omegaconf import OmegaConf

from verl.trainer.benchmarks import math


def test_evaluate_math_response_marks_missing_answer():
    result = math.evaluate_math_response(
        solution="I am not sure.",
        ground_truth="42",
    )

    assert result["pred"] == ""
    assert result["acc"] == 0.0
    assert result["no_answer"] == 1.0


def test_flatten_math_benchmark_metrics_adds_alias_metrics():
    flattened = math.flatten_math_benchmark_metrics(
        {
            "math/math500": {
                "acc": {
                    "mean@4": 0.5,
                    "maj@4/mean": 0.75,
                },
                "no_answer": {
                    "mean@4": 0.25,
                },
            }
        }
    )

    assert flattened["benchmark/math/math500/acc"] == 0.5
    assert flattened["benchmark/math/math500/acc_majority"] == 0.75
    assert flattened["benchmark/math/math500/no_answer"] == 0.25


def test_build_benchmark_runners_supports_all_math_benchmarks(monkeypatch):
    fake_example = math.MathExample(
        uid="math500:0",
        prompt=[{"role": "user", "content": "prompt"}],
        extra_info={"data_source": "math/test"},
        reward_model={"ground_truth": "42", "scoring_module": "math", "info": {}},
    )

    monkeypatch.setattr(math, "_load_examples", lambda *_args, **_kwargs: [fake_example])

    config = OmegaConf.create(
        {
            "trainer": {
                "benchmarks": {
                    "enabled": ["math500", "gsm8k", "aime24", "aime25", "amc23"],
                    "batch_size": 8,
                    "max_samples": 16,
                    "math": {
                        "num_generations": 2,
                        "math500": {},
                        "gsm8k": {},
                        "aime24": {},
                        "aime25": {},
                        "amc23": {},
                    },
                }
            },
            "actor_rollout_ref": {
                "rollout": {
                    "val_kwargs": {"n": 4},
                    "prompt_length": 1024,
                }
            },
        }
    )

    runners = math.build_benchmark_runners(config)

    assert [runner.name for runner in runners] == ["math500", "gsm8k", "aime24", "aime25", "amc23"]
    assert all(runner.examples == [fake_example] for runner in runners)
    assert all(runner.batch_size == 8 for runner in runners)
    assert all(runner.num_generations == 2 for runner in runners)


def test_build_benchmark_runners_uses_benchmark_level_defaults(monkeypatch):
    fake_example = math.MathExample(
        uid="math500:0",
        prompt=[{"role": "user", "content": "prompt"}],
        extra_info={"data_source": "math/test"},
        reward_model={"ground_truth": "42", "scoring_module": "math", "info": {}},
    )

    monkeypatch.setattr(math, "_load_examples", lambda *_args, **_kwargs: [fake_example])

    config = OmegaConf.create(
        {
            "trainer": {
                "benchmarks": {
                    "enabled": ["math500", "gsm8k"],
                    "batch_size": 8,
                    "max_samples": 16,
                    "math": {
                        "math500": {},
                        "gsm8k": {"prompt_length": 2048, "num_generations": 3},
                    },
                }
            },
            "actor_rollout_ref": {
                "rollout": {
                    "val_kwargs": {"n": 4},
                    "prompt_length": 1024,
                }
            },
        }
    )

    runners = math.build_benchmark_runners(config)

    assert [runner.name for runner in runners] == ["math500", "gsm8k"]
    assert [runner.num_generations for runner in runners] == [4, 3]
    assert [runner.prompt_length for runner in runners] == [1024, 2048]
