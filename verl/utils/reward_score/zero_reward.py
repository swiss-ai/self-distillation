def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    del data_source, solution_str, ground_truth, extra_info
    return {
        "score": 0.0,
        "acc": 0.0,
        "pred": "",
        "incorrect_format": 0,
        "feedback": "",
    }
