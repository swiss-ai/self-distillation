from verl.utils.dataset.schema import (
    canonicalize_extra_info,
    canonicalize_reward_model,
    get_data_source,
)


def test_canonicalize_reward_model_moves_legacy_fields_into_info():
    reward_model = canonicalize_reward_model(
        {"ground_truth": "42", "style": "rule", "choices": ["A", "B"]},
        scoring_module="math",
    )

    assert reward_model["ground_truth"] == "42"
    assert reward_model["scoring_module"] == "math"
    assert reward_model["info"] == {"style": "rule", "choices": ["A", "B"]}


def test_canonicalize_extra_info_preserves_original_data_source():
    extra_info = canonicalize_extra_info({"split": "train"}, data_source="openai/gsm8k")

    assert extra_info == {"split": "train", "data_source": "openai/gsm8k"}
    assert get_data_source(extra_info) == "openai/gsm8k"
