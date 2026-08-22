from omegaconf import OmegaConf

from data.prompts import mcq as mcq_prompts
from data.utils import mcq as mcq_utils
from verl.trainer.benchmarks import mcq
from verl.utils.reward_score.feedback import mcq as mcq_scoring


def test_evaluate_mcq_response_prefers_boxed_answer():
    result = mcq_scoring.evaluate_mcq_response(
        solution="Reasoning... final choice is \\boxed{C}.",
        ground_truth="C",
        allowed_choices=("A", "B", "C", "D"),
    )

    assert result["pred"] == "C"
    assert result["acc"] == 1.0
    assert result["incorrect_format"] == 0.0


def test_evaluate_mcq_response_marks_incorrect_format():
    result = mcq_scoring.evaluate_mcq_response(
        solution="I cannot tell.",
        ground_truth="A",
        allowed_choices=("A", "B", "C", "D"),
    )

    assert result["pred"] is None
    assert result["acc"] == 0.0
    assert result["incorrect_format"] == 1.0


def test_evaluate_mcq_response_supports_extended_choice_sets():
    result = mcq_scoring.evaluate_mcq_response(
        solution="After checking each option carefully, the answer is (H).",
        ground_truth="H",
        allowed_choices=tuple("ABCDEFGHIJ"),
    )

    assert result["pred"] == "H"
    assert result["acc"] == 1.0
    assert result["incorrect_format"] == 0.0


def test_flatten_benchmark_metrics_adds_alias_metrics():
    flattened = mcq_scoring.flatten_benchmark_metrics(
        {
            "mcq/gpqa_diamond": {
                "acc": {
                    "mean@4": 0.5,
                    "maj@4/mean": 0.75,
                },
                "incorrect_format": {
                    "mean@4": 0.25,
                },
            }
        }
    )

    assert flattened["benchmark/mcq/gpqa_diamond/acc"] == 0.5
    assert flattened["benchmark/mcq/gpqa_diamond/acc_majority"] == 0.75
    assert flattened["benchmark/mcq/gpqa_diamond/incorrect_format"] == 0.25


def test_compute_score_uses_reward_model_info_for_flexible_extraction():
    result = mcq_scoring.compute_score(
        solution="After checking each option carefully, the answer is (D).",
        ground_truth="D",
        extra_info={
            "reward_model_info": {
                "allowed_choices": ("A", "B", "C", "D"),
                "answer_extraction": "flexible",
            }
        },
    )

    assert result["pred"] == "D"
    assert result["acc"] == 1.0
    assert result["incorrect_format"] == 0.0


def test_build_mmlu_zero_shot_prompt_matches_eleuther_style():
    prompt = mcq_prompts.build_mmlu_zero_shot_prompt(
        subject="college_computer_science",
        question="What does CPU stand for?",
        options=["Central Processing Unit", "Computer Personal Unit", "Central Program Utility", "Control Power Unit"],
    )

    assert "about college computer science." in prompt
    assert "What does CPU stand for?" in prompt
    assert prompt.rstrip().endswith("Answer:")
    assert "2 + 2 = ?" not in prompt


def test_mmlu_metric_paths_include_overall_domain_and_subject():
    assert mcq_utils.get_mmlu_metric_paths("abstract_algebra") == (
        "mcq/mmlu",
        "mcq/mmlu/stem",
        "mcq/mmlu/stem/abstract_algebra",
    )


def test_build_mmlu_pro_zero_shot_prompt_is_zero_shot():
    prompt = mcq_prompts.build_mmlu_pro_zero_shot_prompt(
        question="Which option is correct?",
        options=["one", "two", "three", "four", "five"],
    )

    assert "Options are:" in prompt
    assert "(E): five" in prompt
    assert "Q: Which option is correct?" in prompt
    assert "Warmup question" not in prompt


def test_build_medical_zero_shot_prompt():
    prompt = mcq_prompts.build_medical_zero_shot_prompt(
        question="Which treatment is indicated?",
        options={"A": "Option 1", "B": "Option 2", "C": "Option 3", "D": "Option 4"},
    )

    assert "medical question" in prompt
    assert "(A) Option 1" in prompt
    assert "(D) Option 4" in prompt
    assert 'The correct answer is' in prompt


def test_build_benchmark_runners_supports_all_mcq_benchmarks(monkeypatch):
    fake_example = mcq.MCQExample(
        uid="benchmark:0",
        prompt=[{"role": "user", "content": "prompt"}],
        extra_info={"data_source": "mcq/test", "metric_paths": ()},
        reward_model={
            "ground_truth": "A",
            "scoring_module": "mcq",
            "info": {
                "allowed_choices": ("A", "B", "C", "D"),
                "answer_extraction": "flexible",
            },
        },
    )

    monkeypatch.setattr(mcq, "_load_examples", lambda *_args, **_kwargs: [fake_example])

    config = OmegaConf.create(
        {
            "trainer": {
                "benchmarks": {
                    "enabled": ["gpqa_diamond", "medmcqa", "medqa_en", "mmlu", "mmlu_pro"],
                    "batch_size": 8,
                    "max_samples": 16,
                    "mcq": {
                        "num_generations": 2,
                        "gpqa_diamond": {},
                        "medmcqa": {},
                        "medqa_en": {},
                        "mmlu": {},
                        "mmlu_pro": {},
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

    runners = mcq.build_benchmark_runners(config)

    assert [runner.name for runner in runners] == ["gpqa_diamond", "medmcqa", "medqa_en", "mmlu", "mmlu_pro"]
    assert all(runner.examples == [fake_example] for runner in runners)
    assert all(runner.batch_size == 8 for runner in runners)
    assert all(runner.num_generations == 2 for runner in runners)


def test_build_benchmark_runners_uses_benchmark_level_defaults(monkeypatch):
    fake_example = mcq.MCQExample(
        uid="benchmark:0",
        prompt=[{"role": "user", "content": "prompt"}],
        extra_info={"data_source": "mcq/test", "metric_paths": ()},
        reward_model={
            "ground_truth": "A",
            "scoring_module": "mcq",
            "info": {
                "allowed_choices": ("A", "B", "C", "D"),
                "answer_extraction": "flexible",
            },
        },
    )

    monkeypatch.setattr(mcq, "_load_examples", lambda *_args, **_kwargs: [fake_example])

    config = OmegaConf.create(
        {
            "trainer": {
                "benchmarks": {
                    "enabled": ["medmcqa", "medqa_en"],
                    "batch_size": 8,
                    "max_samples": 16,
                    "num_generations": 3,
                    "prompt_length": 4096,
                    "mcq": {
                        "medmcqa": {},
                        "medqa_en": {"prompt_length": 2048},
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

    runners = mcq.build_benchmark_runners(config)

    assert [runner.name for runner in runners] == ["medmcqa", "medqa_en"]
    assert [runner.num_generations for runner in runners] == [3, 3]
    assert [runner.prompt_length for runner in runners] == [4096, 2048]


def test_build_benchmark_runners_skips_math_benchmarks(monkeypatch):
    fake_example = mcq.MCQExample(
        uid="benchmark:0",
        prompt=[{"role": "user", "content": "prompt"}],
        extra_info={"data_source": "mcq/test", "metric_paths": ()},
        reward_model={
            "ground_truth": "A",
            "scoring_module": "mcq",
            "info": {
                "allowed_choices": ("A", "B", "C", "D"),
                "answer_extraction": "flexible",
            },
        },
    )

    monkeypatch.setattr(mcq, "_load_examples", lambda *_args, **_kwargs: [fake_example])

    config = OmegaConf.create(
        {
            "trainer": {
                "benchmarks": {
                    "enabled": ["gpqa_diamond", "math500"],
                    "batch_size": 8,
                    "max_samples": 16,
                    "mcq": {
                        "gpqa_diamond": {},
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

    runners = mcq.build_benchmark_runners(config)

    assert [runner.name for runner in runners] == ["gpqa_diamond"]
