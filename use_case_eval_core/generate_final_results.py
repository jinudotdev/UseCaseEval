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
        judge_slot = normalize_judge_slot(judge_score)
        indexed_scores.setdefault(join_key, {})
        if judge_slot in indexed_scores[join_key]:
            raise ValueError(
                f"Duplicate {judge_slot} score row for join key: {format_join_key(join_key)}"
            )
        indexed_scores[join_key][judge_slot] = judge_score
    return indexed_scores


def normalize_judge_slot(judge_score):
    judge_slot = (judge_score.get("judge_slot") or "judge_1").strip().lower()
    if judge_slot not in {"judge_1", "judge_2"}:
        raise ValueError(f"Unknown judge slot: {judge_slot}")
    return judge_slot


def format_join_key(join_key):
    return " / ".join(join_key)


def normalize_pass_fail(value):
    normalized = (value or "").strip().lower()
    if normalized in {"pass", "fail"}:
        return normalized
    return ""


def combine_judge_results(judge_1_pass_fail, judge_2_pass_fail):
    judge_1_result = normalize_pass_fail(judge_1_pass_fail)
    judge_2_result = normalize_pass_fail(judge_2_pass_fail)

    if judge_1_result and judge_2_result:
        if judge_1_result == judge_2_result:
            return judge_1_result
        return "mixed"

    return judge_1_result or judge_2_result


def resolve_compatible_context(model_return, judge_scores):
    contexts = []
    model_context = model_return.get("use_case_context", "") or ""
    if model_context:
        contexts.append(("model returns", model_context))

    for judge_slot, judge_score in sorted((judge_scores or {}).items()):
        judge_context = judge_score.get("use_case_context", "") or ""
        if judge_context:
            contexts.append((judge_slot, judge_context))

    if not contexts:
        return ""

    first_source, first_context = contexts[0]
    for source, context in contexts[1:]:
        if context != first_context:
            join_key = make_join_key(model_return)
            raise ValueError(
                "Conflicting use_case_context values for "
                f"{format_join_key(join_key)}: {first_source}={first_context!r}, "
                f"{source}={context!r}"
            )

    return first_context


def build_final_result_row(model_return, judge_scores=None):
    judge_scores = judge_scores or {}
    judge_1_score = judge_scores.get("judge_1", {})
    judge_2_score = judge_scores.get("judge_2", {})
    judge_1_pass_fail = judge_1_score.get("judge_pass_fail", "")
    judge_2_pass_fail = judge_2_score.get("judge_pass_fail", "")
    use_case_context = resolve_compatible_context(model_return, judge_scores)

    return {
        "test_id": model_return["test_id"],
        "use_case": model_return["use_case"],
        "use_case_context": use_case_context,
        "input": model_return["input"],
        "model_name": model_return["model_name"],
        "model_response": model_return["model_response"],
        "tokens_per_second": model_return["tokens_per_second"],
        "judge_1_model": judge_1_score.get("judge_model", ""),
        "judge_1_score": judge_1_score.get("judge_score", ""),
        "judge_1_reason": judge_1_score.get("judge_reason", ""),
        "judge_1_pass_fail": judge_1_pass_fail,
        "judge_2_model": judge_2_score.get("judge_model", ""),
        "judge_2_score": judge_2_score.get("judge_score", ""),
        "judge_2_reason": judge_2_score.get("judge_reason", ""),
        "judge_2_pass_fail": judge_2_pass_fail,
        "combined_judge_result": combine_judge_results(
            judge_1_pass_fail,
            judge_2_pass_fail,
        ),
        "human_score": "",
        "human_notes": "",
    }


def build_final_result_rows(model_returns, judge_scores):
    judge_scores_by_key = index_judge_scores(judge_scores)
    rows = []

    for model_return in model_returns:
        join_key = make_join_key(model_return)
        judge_scores_for_row = judge_scores_by_key.get(join_key)
        if judge_scores_for_row is None or not any(
            judge_slot in judge_scores_for_row for judge_slot in ("judge_1", "judge_2")
        ):
            raise ValueError(f"Missing judge score row for join key: {format_join_key(join_key)}")
        rows.append(
            build_final_result_row(
                model_return,
                judge_scores_for_row,
            )
        )

    return rows
