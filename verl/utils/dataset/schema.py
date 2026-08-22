from __future__ import annotations

from typing import Any, Mapping

REWARD_MODEL_CORE_KEYS = frozenset({"ground_truth", "scoring_module", "info"})


def canonicalize_extra_info(
    extra_info: Mapping[str, Any] | None,
    *,
    data_source: str | None = None,
) -> dict[str, Any]:
    normalized = dict(extra_info or {})
    if data_source is not None and "data_source" not in normalized:
        normalized["data_source"] = data_source
    return normalized


def canonicalize_reward_model(
    reward_model: Mapping[str, Any] | None,
    *,
    scoring_module: str | None = None,
) -> dict[str, Any]:
    normalized = dict(reward_model or {})
    info = dict(normalized.get("info") or {})

    for key, value in normalized.items():
        if key not in REWARD_MODEL_CORE_KEYS:
            info.setdefault(key, value)

    return {
        "ground_truth": normalized.get("ground_truth"),
        "scoring_module": normalized.get("scoring_module", scoring_module),
        "info": info,
    }


def get_data_source(
    extra_info: Mapping[str, Any] | None,
    *,
    fallback: str = "unknown",
) -> str:
    if extra_info is None:
        return fallback
    return str(extra_info.get("data_source", fallback))
