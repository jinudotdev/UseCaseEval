import json
import re

from .schemas import OLLAMA_CHAT_URL

try:
    import requests
except ModuleNotFoundError:
    requests = None


def post_chat(payload, timeout=300):
    if requests is None:
        raise RuntimeError("The requests package is not installed.")
    return requests.post(OLLAMA_CHAT_URL, json=payload, timeout=timeout)


def recover_generated_csv_from_thinking(thinking_text):
    lines = str(thinking_text).splitlines()
    for index, line in enumerate(lines):
        lowered = line.lower()
        if "test_id" in lowered and "input" in lowered:
            return "\n".join(lines[index:]).strip()
    return None


def extract_generated_text(response_json):
    message = response_json.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if content and content.strip():
            return content
        thinking = message.get("thinking")
        if thinking:
            recovered_csv = recover_generated_csv_from_thinking(thinking)
            if recovered_csv:
                return recovered_csv

    for field_name in ("response", "content"):
        content = response_json.get(field_name)
        if content and content.strip():
            return content

    raise ValueError("Could not extract generated text from Ollama response.")


def extract_judge_text(response_json):
    if not isinstance(response_json, dict):
        return ""

    message = response_json.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if content and content.strip():
            return content

    for field_name in ("response", "content"):
        content = response_json.get(field_name)
        if content and content.strip():
            return content

    return ""


def extract_first_json_object(text):
    decoder = json.JSONDecoder()
    last_error = None
    for match in re.finditer(r"{", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError as error:
            last_error = error
            continue
        if isinstance(parsed, dict):
            return parsed, None
    if last_error:
        return None, f"{last_error.msg} at position {last_error.pos}"
    return None, "No JSON object found in judge response."
