JOIN_KEY_FIELDS = [
    "test_id",
    "use_case",
    "input",
    "model_name",
]


def make_join_key(row):
    return tuple(row[field] for field in JOIN_KEY_FIELDS)


def index_judge_scores(judge_scores):
    indexed_scores = {}
    for judge_score in judge_scores:
        join_key = make_join_key(judge_score)
        if join_key in indexed_scores:
            raise ValueError(f"Duplicate judge score row for join key: {format_join_key(join_key)}")
        indexed_scores[join_key] = judge_score
    return indexed_scores


def format_join_key(join_key):
    return " / ".join(join_key)


def build_final_result_row(model_return, judge_score=None):
    judge_score = judge_score or {}
    return {
        "test_id": model_return["test_id"],
        "use_case": model_return["use_case"],
        "input": model_return["input"],
        "model_name": model_return["model_name"],
        "model_response": model_return["model_response"],
        "latency_ms": model_return["latency_ms"],
        "tokens_per_second": model_return["tokens_per_second"],
        "judge_1_model": judge_score.get("judge_model", ""),
        "judge_1_score": judge_score.get("judge_score", ""),
        "judge_1_reason": judge_score.get("judge_reason", ""),
        "judge_1_pass_fail": judge_score.get("judge_pass_fail", ""),
        "judge_2_model": "",
        "judge_2_score": "",
        "judge_2_reason": "",
        "judge_2_pass_fail": "",
        "human_score": "",
        "human_notes": "",
    }


def build_final_result_rows(model_returns, judge_scores):
    judge_scores_by_key = index_judge_scores(judge_scores)
    rows = []

    for model_return in model_returns:
        join_key = make_join_key(model_return)
        judge_score = judge_scores_by_key.get(join_key)
        if judge_score is None:
            raise ValueError(f"Missing judge score row for join key: {format_join_key(join_key)}")
        rows.append(
            build_final_result_row(
                model_return,
                judge_score,
            )
        )

    return rows
