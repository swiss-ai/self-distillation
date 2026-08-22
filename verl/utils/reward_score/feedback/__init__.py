from verl.utils.reward_score.feedback import llm, math, mcq, zero

SUPPORTED_FEEDBACK_SCORING_MODULES = frozenset({"llm", "math", "mcq", "zero"})


def compute_score(
    *,
    solution_str: str,
    ground_truth: str,
    scoring_module: str,
    extra_info: dict | None = None,
    judge_prompt_template: str | None = None,
) -> dict:
    if scoring_module == "math":
        return math.compute_score(solution_str, ground_truth, extra_info)
    if scoring_module == "mcq":
        return mcq.compute_score(solution_str, ground_truth, extra_info)
    if scoring_module == "zero":
        return zero.compute_score(solution_str, ground_truth, extra_info)
    if scoring_module == "llm":
        return llm.compute_score(
            solution_str,
            ground_truth,
            extra_info,
            judge_prompt_template=judge_prompt_template,
        )
    raise ValueError(
        f"Unsupported feedback scoring_module {scoring_module!r}. "
        f"Supported modules: {sorted(SUPPORTED_FEEDBACK_SCORING_MODULES)}."
    )
