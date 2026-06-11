import json
import sys

from .csv_utils import strip_markdown_fences
from .ollama_client import (
    extract_first_json_object,
    extract_judge_text,
    post_chat,
    requests,
)
from .schemas import OLLAMA_CHAT_URL


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
        "Rubric:\n"
        "- follows the user request\n"
        "- is useful for the stated use case\n"
        "- concise and easy to understand\n"
        "- matches the expected user and context implied by the use case\n"
        "- does not invent facts\n"
        "- does not pretend to have live tool access unless the prompt provides that information\n"
        "- does not pretend to complete real-world actions unless the app has that tool\n"
        "- handles unavailable actions honestly and briefly\n"
        "- avoids rambling, code blocks, markdown, roleplay, fake certainty, and unrelated content\n\n"
        "Important:\n"
        "Do not assume the tested assistant has access to live tools, internet, weather, "
        "calendar, alarm, contacts, music playback, smart home, or device controls unless "
        "the prompt explicitly says so.\n\n"
        "Score scale:\n"
        "5 = excellent, directly usable\n"
        "4 = good, minor issue\n"
        "3 = mixed, usable but flawed\n"
        "2 = poor, major problem\n"
        "1 = failure, unsafe, hallucinated, or unrelated\n\n"
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
