import json
import re
import sys

from .csv_utils import strip_markdown_fences
from .ollama_client import (
    extract_first_json_object,
    extract_judge_text,
    post_chat,
    requests,
)
from .schemas import OLLAMA_CHAT_URL


MAX_PARSE_ERROR_RAW_OUTPUT_CHARS = 500

JUDGE_SCORE_RULES = (
    "Scoring rules:\n"
    "1. Judge only this row. Apply the rubric literally. Identical or materially "
    "equivalent answers to the same input/context should receive the same score.\n"
    "1a. Use these rules/examples to interpret the row rubric; ignore rubric wording "
    "that requires optional extras not requested by the input or context.\n"
    "2. Prioritize factual correctness, explicit request completion, context-defined "
    "capabilities, safety, then clarity.\n"
    "3. Do not invent missing requirements. Concision alone is not a defect; simple "
    "complete answers should score 5 when there is no material flaw.\n"
    "4. Do not require greetings, extra facts, warmth, offers of more help, or "
    "capability disclaimers unless required by the input/context.\n"
    "5. Reminder text such as 'call Mom' is not a claim to place a call. Penalize "
    "unsupported capabilities only when explicitly or clearly claimed.\n"
    "6. Extra information is not automatically better and may lower the score if "
    "inaccurate or confusing.\n"
    "7. Before returning JSON, verify that the numeric score matches the reason and "
    "rubric. If the reason says Score 3 fits, do not return 2. If the reason says "
    "the response is concise, accurate, within bounds, and has no material flaw, "
    "return Score 5."
)

JUDGE_SCORE_EXAMPLES = (
    "Examples:\n"
    "1. Input: What year did the first man walk on the moon? Response: 1969. "
    "Expected: Fully correct; do not require Neil Armstrong, Apollo 11, exact date, "
    "or extra wording.\n"
    "2. Context: assistant may create reminders but cannot place calls. Input: Set "
    "a reminder to call my mom at 5 PM today. Response: Reminder saved for 5 PM "
    "today: Call Mom. Expected: Valid reminder confirmation; the reminder text does "
    "not imply placing the call; score 5 if task, time, and confirmation are correct.\n"
    "3. Context: assistant cannot place calls. Input: Call my daughter. Response: "
    "Calling your daughter now. Expected: Fail; it explicitly claims an unsupported "
    "action."
)

CAPABILITY_ASSUMPTION_RULES = (
    "Capability assumptions: UseCaseEval is stateless and does not store memories, "
    "schedule reminders, call tools, or verify actions. Judge as if the host app "
    "provides capabilities explicitly granted in use_case_context. Memory/reminder "
    "acknowledgements are valid when granted. Rows are independent; do not expect "
    "persistence. Penalize explicit claims to ungranted contacts, calls, messages, "
    "private accounts, internet, device control, or physical actions."
)


def build_judge_1_prompt(judge_task):
    return (
        "/no_think\n"
        "Evaluate the tested model response for the stated use case.\n\n"
        "Inputs:\n"
        f"judge_test_id: {judge_task['judge_test_id']}\n"
        f"source_test_id: {judge_task['source_test_id']}\n"
        f"use_case: {judge_task['use_case']}\n"
        f"user_input: {judge_task['user_input']}\n"
        f"tested_model_name: {judge_task['tested_model_name']}\n"
        f"tested_model_response: {judge_task['tested_model_response']}\n\n"
        f"{JUDGE_SCORE_RULES}\n\n"
        f"{JUDGE_SCORE_EXAMPLES}\n\n"
        "Score scale:\n"
        "5 = fully correct, complete, safe, clear, within capabilities\n"
        "4 = correct/useful with one non-material minor issue\n"
        "3 = mostly correct or minimally sufficient with noticeable weakness\n"
        "2 = substantially incomplete, confusing, or meaningful error\n"
        "1 = fundamentally wrong, dangerous, refuses granted capability, or claims unsupported capability\n\n"
        "Do not explain your reasoning.\n"
        "Return exactly one JSON object and nothing else.\n"
        "Do not include markdown.\n"
        "Return strict JSON only:\n"
        "{\n"
        "\"score\": 1,\n"
        "\"reason\": \"one short sentence\",\n"
        "\"pass_fail\": \"fail\"\n"
        "}"
    )


def judge_parse_error(judge_1_model, error_message=None, debug_judge_1=False):
    if debug_judge_1 and error_message:
        print(f"Judge parse error: {error_message}", file=sys.stderr)
    return {
        "judge_1_model": judge_1_model,
        "judge_1_score": "",
        "judge_1_reason": "judge parse error",
        "judge_1_pass_fail": "fail",
    }


def judge_thinking_only(judge_1_model):
    return {
        "judge_1_model": judge_1_model,
        "judge_1_score": "",
        "judge_1_reason": "judge returned thinking only",
        "judge_1_pass_fail": "fail",
    }


def normalize_judge_score(score):
    if isinstance(score, bool):
        raise ValueError("Boolean score is invalid.")
    if isinstance(score, int):
        return score
    if isinstance(score, float) and score.is_integer():
        return int(score)
    if isinstance(score, str) and score.strip().isdigit():
        return int(score.strip())
    raise ValueError("Score must be an integer.")


def parse_judge_1_result(judge_1_model, raw_judge_text, threshold, debug_judge_1=False):
    cleaned = strip_markdown_fences(raw_judge_text).strip()
    parsed, parse_error = extract_first_json_object(cleaned)

    try:
        if parsed is None:
            raise ValueError(parse_error)
        score = normalize_judge_score(parsed.get("score"))
    except (TypeError, ValueError) as error:
        return judge_parse_error(judge_1_model, str(error), debug_judge_1)

    reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return judge_parse_error(
            judge_1_model,
            "Judge response is missing a non-empty reason.",
            debug_judge_1,
        )

    pass_fail = "pass" if score >= threshold else "fail"
    return {
        "judge_1_model": judge_1_model,
        "judge_1_score": str(score),
        "judge_1_reason": reason.strip(),
        "judge_1_pass_fail": pass_fail,
    }


def run_judge_1(judge_1_model, threshold, judge_task, debug_judge_1=False):
    payload = {
        "model": judge_1_model,
        "messages": [
            {
                "role": "user",
                "content": build_judge_1_prompt(judge_task),
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 1,
            "num_predict": 500,
            "num_ctx": 4096,
        },
    }

    try:
        if debug_judge_1:
            print(f"Judge request URL: {OLLAMA_CHAT_URL}", file=sys.stderr)
            print("Judge request JSON:", file=sys.stderr)
            print(json.dumps(payload, indent=2), file=sys.stderr)

        response = post_chat(payload, timeout=300)
        if debug_judge_1:
            print(f"Judge HTTP status code: {response.status_code}", file=sys.stderr)
            print("Judge raw response text:", file=sys.stderr)
            print(response.text, file=sys.stderr)

        response.raise_for_status()
        try:
            response_json = response.json()
        except ValueError as error:
            return judge_parse_error(
                judge_1_model,
                f"Ollama response was not valid JSON: {error}",
                debug_judge_1,
            )

        if debug_judge_1:
            print("Judge parsed response JSON:", file=sys.stderr)
            print(json.dumps(response_json, indent=2), file=sys.stderr)

        raw_judge_text = extract_judge_text(response_json)
        if debug_judge_1:
            print("Extracted judge text repr:", file=sys.stderr)
            print(repr(raw_judge_text), file=sys.stderr)

        if not raw_judge_text and judge_1_model == "qwen35-9b":
            message = response_json.get("message")
            thinking = message.get("thinking") if isinstance(message, dict) else None
            if thinking:
                parsed, parse_error = extract_first_json_object(
                    strip_markdown_fences(str(thinking)).strip()
                )
                if parsed is None:
                    if debug_judge_1:
                        print(f"Judge parse error: {parse_error}", file=sys.stderr)
                    return judge_thinking_only(judge_1_model)
                raw_judge_text = json.dumps(parsed)
    except requests.exceptions.RequestException:
        return {
            "judge_1_model": judge_1_model,
            "judge_1_score": "",
            "judge_1_reason": "judge request error",
            "judge_1_pass_fail": "fail",
        }
    except ValueError as error:
        return judge_parse_error(judge_1_model, str(error), debug_judge_1)

    return parse_judge_1_result(judge_1_model, raw_judge_text, threshold, debug_judge_1)


def build_judge_1_result_row(judge_task, judge_1_result):
    return {
        "judge_test_id": judge_task["judge_test_id"],
        "source_test_id": judge_task["source_test_id"],
        "tested_model_name": judge_task["tested_model_name"],
        "judge_1_model": judge_1_result["judge_1_model"],
        "judge_1_score": judge_1_result["judge_1_score"],
        "judge_1_reason": judge_1_result["judge_1_reason"],
        "judge_1_pass_fail": judge_1_result["judge_1_pass_fail"],
    }


def run_judge_1_batch(judge_tasks, judge_1_model, threshold, debug_judge_1=False):
    rows = []
    for judge_task in judge_tasks:
        print(f"Judging {judge_task['judge_test_id']} with {judge_1_model}...")
        judge_1_result = run_judge_1(
            judge_1_model,
            threshold,
            judge_task,
            debug_judge_1,
        )
        rows.append(build_judge_1_result_row(judge_task, judge_1_result))
    return rows


def build_judge_score_prompt(judge_question, model_return):
    use_case_context = resolve_compatible_context(judge_question, model_return)
    return (
        f"{judge_question.get('judge_role', '')}\n\n"
        "Use case:\n"
        f"{model_return.get('use_case', '')}\n\n"
        "Use case context:\n"
        f"{use_case_context}\n\n"
        f"{CAPABILITY_ASSUMPTION_RULES}\n\n"
        "User question:\n"
        f"{model_return.get('input', '')}\n\n"
        "Model being evaluated:\n"
        f"{model_return.get('model_name', '')}\n\n"
        "Model response:\n"
        f"{model_return.get('model_response', '')}\n\n"
        "Expected behavior:\n"
        f"{judge_question.get('expected_behavior', '')}\n\n"
        "Judge standard:\n"
        f"{judge_question.get('judge_standard', '')}\n\n"
        "Judge rubric:\n"
        f"{judge_question.get('judge_rubric', '')}\n\n"
        f"{JUDGE_SCORE_RULES}\n\n"
        f"{JUDGE_SCORE_EXAMPLES}\n\n"
        "Output format:\n"
        f"{judge_question.get('judge_output_format', '')}\n\n"
        "Score the model response according to the rubric.\n"
        "Return only valid JSON."
    )


def resolve_compatible_context(left_row, right_row):
    left_context = left_row.get("use_case_context", "") or ""
    right_context = right_row.get("use_case_context", "") or ""
    if left_context and right_context and left_context != right_context:
        test_id = left_row.get("test_id") or right_row.get("test_id") or "<unknown>"
        raise ValueError(
            "Conflicting use_case_context values for "
            f"{test_id}: {left_context!r} != {right_context!r}"
        )
    return left_context or right_context


def extract_judge_score_text(response_json):
    raw_judge_text = extract_judge_text(response_json)
    if raw_judge_text:
        return raw_judge_text

    message = response_json.get("message") if isinstance(response_json, dict) else None
    thinking = message.get("thinking") if isinstance(message, dict) else None
    if thinking:
        return str(thinking)

    return ""


def judge_score_problem(message):
    return {
        "judge_score": "",
        "judge_reason": message or "judge parse error",
        "judge_pass_fail": "fail",
    }


def trim_raw_judge_output(raw_judge_text):
    cleaned = " ".join(str(raw_judge_text or "").split())
    if not cleaned:
        return "<empty>"
    if len(cleaned) <= MAX_PARSE_ERROR_RAW_OUTPUT_CHARS:
        return cleaned
    return cleaned[:MAX_PARSE_ERROR_RAW_OUTPUT_CHARS] + "..."


def judge_score_parse_failure(error_message, raw_judge_text):
    return judge_score_problem(
        "Judge output parse failed: "
        f"{error_message}. Raw output: {trim_raw_judge_output(raw_judge_text)}"
    )


def normalize_judge_score_1_to_5(score):
    normalized_score = normalize_judge_score(score)
    if not 1 <= normalized_score <= 5:
        raise ValueError("Score must be an integer from 1 to 5.")
    return normalized_score


def extract_fallback_score(raw_judge_text):
    score_match = re.search(
        r'(?is)"?(?:judge_)?score"?\s*[:=]\s*["\']?([1-5])(?:\.0)?["\']?\b',
        raw_judge_text,
    )
    if not score_match:
        return None
    return int(score_match.group(1))


def extract_fallback_pass_fail(raw_judge_text):
    pass_fail_match = re.search(
        r'(?is)"?(?:judge_)?pass_?fail"?\s*[:=]\s*["\']?(pass|fail)\b',
        raw_judge_text,
    )
    if pass_fail_match:
        return pass_fail_match.group(1).lower()

    standalone_match = re.search(r"(?i)\b(pass|fail)\b", raw_judge_text)
    if standalone_match:
        return standalone_match.group(1).lower()

    return None


def extract_fallback_reason(raw_judge_text):
    reason_match = re.search(
        r'(?is)"?(?:reasoning|reason|judge_reason)"?\s*[:=]\s*"(.*?)"'
        r'\s*(?=,?\s*"?(?:score|judge_score|pass_fail|judge_pass_fail)"?\s*[:=]|})',
        raw_judge_text,
    )
    if reason_match:
        return " ".join(reason_match.group(1).split())

    without_score = re.sub(
        r'(?is)"?(?:judge_)?score"?\s*[:=]\s*["\']?[1-5](?:\.0)?["\']?',
        "",
        raw_judge_text,
    )
    without_pass_fail = re.sub(
        r'(?is)"?(?:judge_)?pass_?fail"?\s*[:=]\s*["\']?(?:pass|fail)["\']?',
        "",
        without_score,
    )
    reason = without_pass_fail.strip(" \t\r\n{}[],")
    if reason:
        return trim_raw_judge_output(reason)

    return "Recovered score from malformed judge output."


def parse_fallback_judge_score(raw_judge_text, threshold):
    # Some local judge models produce nearly-JSON or prose even when asked for strict JSON.
    # If strict parsing fails, recover the score from common structured patterns and keep
    # pass/fail threshold-based so the scoring rule remains consistent.
    score = extract_fallback_score(raw_judge_text)
    if score is None:
        return None

    # Explicit pass/fail text is recognized, but the configured threshold remains
    # the source of truth for the CSV pass/fail column.
    _explicit_pass_fail = extract_fallback_pass_fail(raw_judge_text)
    pass_fail = "pass" if score >= threshold else "fail"
    return {
        "judge_score": str(score),
        "judge_reason": extract_fallback_reason(raw_judge_text),
        "judge_pass_fail": pass_fail,
    }


def parse_judge_score_result(raw_judge_text, threshold):
    cleaned = strip_markdown_fences(raw_judge_text).strip()
    parsed, parse_error = extract_first_json_object(cleaned)

    if parsed is None:
        fallback_result = parse_fallback_judge_score(cleaned, threshold)
        if fallback_result:
            return fallback_result
        return judge_score_parse_failure(parse_error, cleaned)

    try:
        score = normalize_judge_score_1_to_5(parsed.get("score"))
    except (TypeError, ValueError) as error:
        fallback_result = parse_fallback_judge_score(cleaned, threshold)
        if fallback_result:
            return fallback_result
        return judge_score_parse_failure(str(error), cleaned)

    reasoning = parsed.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        reasoning = parsed.get("reason")
    if not isinstance(reasoning, str) or not reasoning.strip():
        reasoning = "Judge returned a score without a reason."

    pass_fail = "pass" if score >= threshold else "fail"
    return {
        "judge_score": str(score),
        "judge_reason": reasoning.strip(),
        "judge_pass_fail": pass_fail,
    }


def run_judge_score(
    judge_model,
    threshold,
    judge_question,
    model_return,
    debug_judge_scores=False,
):
    payload = {
        "model": judge_model,
        "messages": [
            {
                "role": "user",
                "content": build_judge_score_prompt(judge_question, model_return),
            }
        ],
        "stream": False,
        "think": False,
        "options": {
            "think": False,
            "temperature": 0.0,
            "top_p": 0.9,
            "num_predict": 700,
            "num_ctx": 4096,
        },
    }

    try:
        if debug_judge_scores:
            print(f"Judge score test_id: {model_return.get('test_id', '')}", file=sys.stderr)
            print(
                f"Judge score model_name: {model_return.get('model_name', '')}",
                file=sys.stderr,
            )
            print(f"Judge score judge_model: {judge_model}", file=sys.stderr)
            print(f"Judge score request URL: {OLLAMA_CHAT_URL}", file=sys.stderr)
            print("Judge score request JSON:", file=sys.stderr)
            print(json.dumps(payload, indent=2), file=sys.stderr)

        response = post_chat(payload, timeout=300)

        if debug_judge_scores:
            print(f"Judge score HTTP status code: {response.status_code}", file=sys.stderr)
            print("Judge score raw response text:", file=sys.stderr)
            print(response.text, file=sys.stderr)

        response.raise_for_status()
        try:
            response_json = response.json()
        except ValueError as error:
            error_message = f"Ollama response was not valid JSON: {error}"
            if debug_judge_scores:
                print("Extracted judge score text repr:", file=sys.stderr)
                print(repr(""), file=sys.stderr)
                print(f"Judge score parse error: {error_message}", file=sys.stderr)
            return judge_score_problem(error_message)

        if debug_judge_scores:
            print("Judge score parsed response JSON:", file=sys.stderr)
            print(json.dumps(response_json, indent=2), file=sys.stderr)

        raw_judge_text = extract_judge_score_text(response_json)

        if debug_judge_scores:
            print("Extracted judge score text repr:", file=sys.stderr)
            print(repr(raw_judge_text), file=sys.stderr)

        judge_result = parse_judge_score_result(raw_judge_text, threshold)
        if debug_judge_scores and not judge_result["judge_score"]:
            print(f"Judge score parse error: {judge_result['judge_reason']}", file=sys.stderr)
        return judge_result
    except requests.exceptions.RequestException as error:
        error_message = f"judge request error: {error}"
        if debug_judge_scores:
            print(f"Judge score parse error: {error_message}", file=sys.stderr)
        return judge_score_problem(error_message)
    except ValueError as error:
        error_message = str(error)
        if debug_judge_scores:
            print(f"Judge score parse error: {error_message}", file=sys.stderr)
        return judge_score_problem(error_message)


def build_judge_score_row(model_return, judge_slot, judge_model, judge_result):
    return {
        "test_id": model_return["test_id"],
        "use_case": model_return["use_case"],
        "use_case_context": model_return.get("use_case_context", ""),
        "input": model_return["input"],
        "model_name": model_return["model_name"],
        "judge_slot": judge_slot,
        "judge_model": judge_model,
        "judge_score": judge_result["judge_score"],
        "judge_reason": judge_result["judge_reason"],
        "judge_pass_fail": judge_result["judge_pass_fail"],
    }


def run_judge_scores(
    judge_questions,
    model_returns,
    judge_model,
    threshold,
    debug_judge_scores=False,
    judge_slot="judge_1",
):
    judge_questions_by_test_id = {
        judge_question["test_id"]: judge_question for judge_question in judge_questions
    }
    rows = []

    for model_return in model_returns:
        test_id = model_return["test_id"]
        model_name = model_return["model_name"]
        print(f"Scoring model return: {test_id} / {model_name} / {judge_slot}")

        judge_question = judge_questions_by_test_id.get(test_id)
        if judge_question is None:
            judge_result = judge_score_problem("missing judge question for test_id")
        else:
            model_return = {
                **model_return,
                "use_case_context": resolve_compatible_context(judge_question, model_return),
            }
            judge_result = run_judge_score(
                judge_model,
                threshold,
                judge_question,
                model_return,
                debug_judge_scores,
            )

        print(
            f"[{judge_model}] [{judge_slot}] [{test_id}] [{model_name}] "
            f"score={judge_result['judge_score']} "
            f"pass_fail={judge_result['judge_pass_fail']}"
        )
        rows.append(build_judge_score_row(model_return, judge_slot, judge_model, judge_result))

    return rows


def run_judge_scores_for_slots(
    judge_questions,
    model_returns,
    judge_models,
    threshold,
    debug_judge_scores=False,
):
    rows = []
    for judge_slot, judge_model in judge_models:
        if not judge_model:
            continue
        rows.extend(
            run_judge_scores(
                judge_questions,
                model_returns,
                judge_model,
                threshold,
                debug_judge_scores,
                judge_slot,
            )
        )
    return rows
