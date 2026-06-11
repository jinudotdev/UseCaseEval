import json
import sys

from .ollama_client import extract_generated_text, post_chat
from .schemas import OLLAMA_CHAT_URL


def build_generator_prompt(use_case, num_tests):
    return (
        "You are creating an evaluation set for local LLM models.\n"
        f"Use case: {use_case}\n"
        f"Create exactly {num_tests} test prompts.\n"
        f"Return exactly {num_tests + 1} CSV lines total: one header line plus {num_tests} data rows.\n"
        "Each prompt should test one realistic task for that use case.\n"
        "Each input should be under 20 words.\n"
        "Prompts should be short, practical, and varied.\n"
        "Do not include answers.\n"
        "Do not use markdown.\n"
        "Do not wrap the CSV in code fences.\n"
        "Do not add commentary before or after the CSV.\n"
        "The first line must be exactly: test_id,input\n"
        "Return only valid CSV with columns:\n"
        "test_id,input"
    )


def generate_tests_csv(generator_model, use_case, num_tests, debug_generator=False):
    payload = {
        "model": generator_model,
        "think": False,
        "messages": [
            {
                "role": "user",
                "content": "/no_think\n" + build_generator_prompt(use_case, num_tests),
            }
        ],
        "stream": False,
        "options": {
            "think": False,
            "temperature": 0.4,
            "top_p": 0.9,
            "num_predict": 400,
            "num_ctx": 4096,
        },
    }

    if debug_generator:
        print(f"Generator request URL: {OLLAMA_CHAT_URL}", file=sys.stderr)
        print("Generator request JSON:", file=sys.stderr)
        print(json.dumps(payload, indent=2), file=sys.stderr)

    response = post_chat(payload, timeout=300)

    if debug_generator:
        print(f"Generator HTTP status code: {response.status_code}", file=sys.stderr)
        print("Generator raw response text:", file=sys.stderr)
        print(response.text, file=sys.stderr)

    response.raise_for_status()
    data = response.json()

    if debug_generator:
        print("Generator parsed response JSON:", file=sys.stderr)
        print(json.dumps(data, indent=2), file=sys.stderr)

    if data.get("done_reason") == "length":
        print("Warning: generator hit token limit. Attempting CSV recovery.", file=sys.stderr)

    generated_text = extract_generated_text(data)

    if debug_generator:
        print("Extracted generated text repr:", file=sys.stderr)
        print(repr(generated_text), file=sys.stderr)

    return generated_text


def prompt_for_use_case():
    while True:
        use_case = input("What use case do you want to test? ").strip()
        if use_case:
            return use_case
        print("Please enter a use case.")


def prompt_for_num_tests(default_num_tests=10):
    while True:
        raw_value = input(f"How many questions should I generate? [{default_num_tests}] ").strip()
        if not raw_value:
            return default_num_tests
        try:
            num_tests = int(raw_value)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if num_tests >= 1:
            return num_tests
        print("Please enter a number greater than 0.")
