def compute_score(
    solution_str: str,
    ground_truth: str | None,
    extra_info: dict | None = None,
) -> dict:
    del solution_str, ground_truth, extra_info
    return {
        "score": 0.0,
        "acc": 0.0,
        "pred": "",
        "incorrect_format": 0,
        "feedback": "",
    }
