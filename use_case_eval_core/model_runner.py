import sys
import time

from .ollama_client import post_chat, requests


SYSTEM_PROMPT = (
    "You are a local mobile assistant. Return only the requested final answer. "
    "Do not say Sure. Do not explain. Do not label the answer. "
    "Do not mention that you are following instructions. Do not include reasoning. "
    "Do not ask follow-up questions. Keep it short and mobile-friendly."
)


def generate_response(model_name, user_input, max_tokens):
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_input,
            },
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "repeat_penalty": 1.15,
            "num_predict": max_tokens,
            "num_ctx": 2048,
        },
    }

    started_at = time.perf_counter()
    response = post_chat(payload, timeout=300)
    latency_ms = (time.perf_counter() - started_at) * 1000
    response.raise_for_status()

    data = response.json()
    tokens_per_second = calculate_tokens_per_second(data)
    return data["message"]["content"], latency_ms, tokens_per_second


def calculate_tokens_per_second(data):
    eval_count = data.get("eval_count")
    eval_duration = data.get("eval_duration")

    if not eval_count or not eval_duration:
        return ""

    eval_duration_seconds = eval_duration / 1_000_000_000
    if eval_duration_seconds <= 0:
        return ""

    return eval_count / eval_duration_seconds


def format_number(value):
    if value == "":
        return ""
    return f"{value:.2f}"


def build_result_row(test, use_case, model_name, model_response, latency_ms, tokens_per_second):
    return {
        "test_id": test["test_id"],
        "use_case": use_case,
        "input": test["input"],
        "model_name": model_name,
        "model_response": model_response,
        "latency_ms": format_number(latency_ms),
        "tokens_per_second": format_number(tokens_per_second),
        "judge_1_model": "",
        "judge_1_score": "",
        "judge_1_reason": "",
        "judge_1_pass_fail": "",
        "judge_2_model": "",
        "judge_2_score": "",
        "judge_2_reason": "",
        "judge_2_pass_fail": "",
        "human_score": "",
        "human_notes": "",
    }


def build_model_return_row(test, use_case, model_name, model_response, latency_ms, tokens_per_second):
    return {
        "test_id": test["test_id"],
        "use_case": use_case,
        "input": test["input"],
        "model_name": model_name,
        "model_response": model_response,
        "latency_ms": format_number(latency_ms),
        "tokens_per_second": format_number(tokens_per_second),
    }


def run_model_returns(use_case, model_names, tests, max_tokens):
    rows = []

    for model_name in model_names:
        print(f"Running model: {model_name}")
        for test in tests:
            row_started_at = time.perf_counter()
            try:
                model_response, latency_ms, tokens_per_second = generate_response(
                    model_name, test["input"], max_tokens
                )
            except requests.exceptions.ConnectionError:
                print(
                    "Error: Could not connect to Ollama at http://localhost:11434. "
                    "Make sure Ollama is running, then try again.",
                    file=sys.stderr,
                )
                return None
            except requests.exceptions.RequestException as error:
                model_response = f"ERROR: {error}"
                latency_ms = (time.perf_counter() - row_started_at) * 1000
                tokens_per_second = ""
            except ValueError as error:
                model_response = f"ERROR: Invalid response from Ollama: {error}"
                latency_ms = (time.perf_counter() - row_started_at) * 1000
                tokens_per_second = ""

            rows.append(
                build_model_return_row(
                    test,
                    use_case,
                    model_name,
                    model_response,
                    latency_ms,
                    tokens_per_second,
                )
            )
            print(f"[{model_name}] [{test['test_id']}] done in {format_number(latency_ms)} ms")

    return rows


def run_eval(use_case, model_names, tests, max_tokens):
    rows = []

    for model_name in model_names:
        print(f"Running model: {model_name}")
        for test in tests:
            row_started_at = time.perf_counter()
            try:
                model_response, latency_ms, tokens_per_second = generate_response(
                    model_name, test["input"], max_tokens
                )
            except requests.exceptions.ConnectionError:
                print(
                    "Error: Could not connect to Ollama at http://localhost:11434. "
                    "Make sure Ollama is running, then try again.",
                    file=sys.stderr,
                )
                return None
            except requests.exceptions.RequestException as error:
                model_response = f"ERROR: {error}"
                latency_ms = (time.perf_counter() - row_started_at) * 1000
                tokens_per_second = ""
            except ValueError as error:
                model_response = f"ERROR: Invalid response from Ollama: {error}"
                latency_ms = (time.perf_counter() - row_started_at) * 1000
                tokens_per_second = ""

            rows.append(
                build_result_row(
                    test,
                    use_case,
                    model_name,
                    model_response,
                    latency_ms,
                    tokens_per_second,
                )
            )
            print(f"[{model_name}] [{test['test_id']}] done in {format_number(latency_ms)} ms")

    return rows
