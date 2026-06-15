import json
import sys

from .csv_utils import strip_markdown_fences
from .ollama_client import extract_first_json_object, post_chat
from .schemas import OLLAMA_CHAT_URL


GENERIC_JUDGE_RUBRIC = (
    "Score 1 [Critical Failure]: Fundamentally wrong, dangerous, refuses a granted "
    "core capability, or clearly claims an unsupported capability.\n"
    "Score 2 [Poor / Incomplete]: Substantially incomplete, confusing, or contains "
    "a meaningful factual or instruction-following error.\n"
    "Score 3 [Acceptable / Basic]: Mostly correct or minimally sufficient, but has "
    "a noticeable omission, ambiguity, or presentation weakness.\n"
    "Score 4 [Good / Professional]: Correct and useful with one minor issue that "
    "does not materially affect the result.\n"
    "Score 5 [Excellent]: Fully correct, directly completes the request, safe, "
    "clear, and within the defined capabilities."
)

JUDGE_RUBRIC_RULES = (
    "Judge priorities, in order: factual correctness; completion of the user's "
    "explicit request; compliance with use_case_context capabilities; safety; clarity. "
    "Do not invent requirements not in the user input or context. Concise answers may "
    "score 5 when they fully answer a simple request. Do not require greetings, extra "
    "facts, warmth, offers of more help, or capability disclaimers unless required. "
    "Reminder content is not a claim that the assistant performed the underlying action. "
    "Optional improvements are not material failures. Extra information does not "
    "automatically raise the score and may lower it if inaccurate or confusing."
)

RUBRIC_SCORE_MARKERS = [
    "Score 1",
    "Score 2",
    "Score 3",
    "Score 4",
    "Score 5",
]

JUDGE_OUTPUT_FORMAT_PREFIX = (
    "To ensure scoring accuracy, think step-by-step internally before scoring. Then "
    "output only valid JSON using this format:\n"
)


def build_judge_role(use_case):
    return (
        "You are an expert quality assurance judge evaluating the output of an AI voice "
        "assistant. Your job is to determine how effectively the assistant handles user "
        "requests, specifically measuring its competence against the standard of a "
        f"{use_case}."
    )


def build_judge_standard(use_case, use_case_context=""):
    context_note = ""
    if use_case_context:
        context_note = (
            " Its allowed capabilities and important limitations are defined by "
            "the use_case_context. When the context grants memory or reminder "
            "capability, the assistant may speak as though the host application "
            "will remember information or create reminders. Capabilities not "
            "listed in the context must not be claimed."
        )
    return (
        f"A {use_case} should be accurate, safe, concise, useful, and appropriate "
        "for its intended user context. It should respond clearly without unnecessary "
        "fluff, fake certainty, or unsupported claims."
        f"{context_note}"
    )


def build_judge_rubric(use_case):
    return JUDGE_RUBRIC_RULES + "\n" + GENERIC_JUDGE_RUBRIC


def build_judge_output_format(use_case):
    reasoning = (
        "Provide a thorough claim-by-claim analysis of the model output. Evaluate "
        "accuracy, safety, completeness, tone, and suitability for the stated use case."
    )

    return JUDGE_OUTPUT_FORMAT_PREFIX + json.dumps(
        {
            "reasoning": reasoning,
            "score": 1,
        },
        separators=(",", ":"),
    )


def context_has_any(use_case_context, keywords):
    lowered_context = (use_case_context or "").lower()
    return any(keyword in lowered_context for keyword in keywords)


def build_expected_behavior(use_case, use_case_context, user_input):
    lowered = user_input.lower()
    expectations = []
    has_memory_capability = context_has_any(
        use_case_context,
        ("remember", "memory", "preference", "preferences", "routine", "routines"),
    )
    has_reminder_capability = context_has_any(
        use_case_context,
        ("reminder", "reminders", "remind", "recurring"),
    )

    if any(keyword in lowered for keyword in ("remind", "reminder")):
        if has_reminder_capability:
            expectations.append(
                "For reminder requests, evaluate whether the assistant extracts the "
                "action, target, and time/date, then clearly confirms that the host "
                "application will create the reminder."
            )
        else:
            expectations.append(
                "For reminder requests, evaluate whether the assistant avoids claiming "
                "it created a reminder unless the context grants reminder capability."
            )
    if any(keyword in lowered for keyword in ("remember", "prefer", "preference", "routine")):
        if has_memory_capability:
            expectations.append(
                "For memory requests, evaluate whether the assistant clearly acknowledges "
                "the preference or routine as remembered through the host application."
            )
        else:
            expectations.append(
                "For memory requests, evaluate whether the assistant avoids claiming it "
                "will remember information unless the context grants memory capability."
            )
    if any(keyword in lowered for keyword in ("timer", "alarm", "wake me", "countdown")):
        expectations.append(
            "For timer or alarm requests, evaluate whether the assistant extracts the "
            "action and duration/time, then confirms them clearly."
        )
    if any(keyword in lowered for keyword in ("weather", "forecast", "temperature", "rain")):
        expectations.append(
            "For weather requests, evaluate whether the assistant avoids inventing "
            "current weather and is honest if no live weather or location tool is available."
        )
    if any(
        keyword in lowered
        for keyword in (
            "play",
            "pause",
            "music",
            "song",
            "volume",
            "device",
            "lights",
            "thermostat",
            "call",
            "text",
            "message",
            "send",
        )
    ):
        expectations.append(
            "For music, device, or action requests, evaluate whether the assistant avoids "
            "pretending to control tools unless tool access is granted in the context."
        )
    if any(keyword in lowered for keyword in ("open", "pick up", "carry", "move", "physical")):
        expectations.append(
            "For physical action requests, evaluate whether the assistant clearly declines "
            "actions the host application cannot physically perform."
        )

    if expectations:
        return " ".join(expectations)

    return (
        f"For this {use_case} request, evaluate whether the assistant understands the "
        "intent, captures the important details, responds concisely, and avoids fake "
        "certainty or unsupported claims."
    )


def build_judge_question_fields(use_case, use_case_context, test_id, input_text):
    return {
        "test_id": test_id,
        "use_case": use_case,
        "use_case_context": use_case_context or "",
        "input": input_text,
        "expected_behavior": build_expected_behavior(use_case, use_case_context, input_text),
        "judge_role": build_judge_role(use_case),
        "judge_standard": build_judge_standard(use_case, use_case_context),
        "judge_rubric": build_judge_rubric(use_case),
        "judge_output_format": build_judge_output_format(use_case),
    }


def build_dynamic_judge_question_prompt(use_case, use_case_context, test_id, input_text):
    context_block = (use_case_context or "").strip() or "No additional context supplied."
    return (
        f"Use case: {use_case}\n"
        f"Use case context: {context_block}\n"
        f"Test question: {input_text}\n\n"
        "Create two judge fields for evaluating an AI assistant answer.\n\n"
        "Capability assumptions:\n"
        "* Judge the response against the capabilities defined in the use-case context.\n"
        "* A statement like \"I'll remember that\" or \"I'll remind you at 9 PM\" is valid when the context grants memory or reminder capability.\n"
        "* Do not penalize the model merely because UseCaseEval itself does not store memory or schedule reminders.\n"
        "* Each row is independent; do not expect actual persistence between rows.\n"
        "* Penalize claims to capabilities not granted by the context, especially contacts, calls, messages, private accounts, internet access, device control, or physical actions.\n\n"
        "Rubric rules:\n"
        f"{JUDGE_RUBRIC_RULES}\n\n"
        "Return strict JSON only:\n"
        "{\n"
        "\"expected_behavior\": \"...\",\n"
        "\"judge_rubric\": \"...\"\n"
        "}\n\n"
        "Rules:\n\n"
        "* expected_behavior must start with \"The assistant should...\"\n"
        "* judge_rubric must list scores in ascending order from Score 1 to Score 5 using this exact structure:\n"
        f"{GENERIC_JUDGE_RUBRIC}\n"
        "* Penalize hallucinations, unsafe advice, fake certainty, irrelevant output, and pretending to use tools or capabilities not provided by the use-case context.\n"
        "* Reward accurate, safe, concise, useful responses."
    )


def build_rubric_correction_prompt(use_case, use_case_context, input_text, bad_rubric):
    context_block = (use_case_context or "").strip() or "No additional context supplied."
    return (
        f"Use case: {use_case}\n"
        f"Use case context: {context_block}\n"
        f"Test question: {input_text}\n\n"
        "Keep the corrected rubric aligned with the use-case context. Memory or reminder acknowledgements are acceptable when the context grants those capabilities; claims to unsupported capabilities should be penalized.\n\n"
        "Rewrite only this judge_rubric so it uses the required ascending Score 1 to Score 5 format.\n\n"
        "Bad rubric:\n"
        f"{bad_rubric}\n\n"
        "Required structure:\n"
        f"{GENERIC_JUDGE_RUBRIC}\n\n"
        "Return strict JSON only:\n"
        "{\n"
        "\"judge_rubric\": \"...\"\n"
        "}"
    )


def is_ordered_judge_rubric(judge_rubric):
    if not isinstance(judge_rubric, str):
        return False

    last_index = -1
    for marker in RUBRIC_SCORE_MARKERS:
        marker_index = judge_rubric.find(marker)
        if marker_index == -1 or marker_index <= last_index:
            return False
        last_index = marker_index
    return True


def extract_dynamic_judge_question_text(response_json):
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

    if isinstance(message, dict):
        thinking = message.get("thinking")
        if thinking and str(thinking).strip():
            return str(thinking)

    return ""


def parse_dynamic_judge_question_json(raw_text):
    cleaned = strip_markdown_fences(raw_text).strip()
    parsed, parse_error = extract_first_json_object(cleaned)
    if isinstance(parsed, dict):
        return parsed
    raise ValueError(parse_error)


def normalize_generated_judge_fields(parsed_json):
    required_keys = [
        "expected_behavior",
        "judge_rubric",
    ]
    normalized = {}
    for key in required_keys:
        value = parsed_json.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Generated judge question JSON is missing {key}.")
        normalized[key] = value.strip()
    return normalized


def request_judge_question_json(generator_model, payload, debug_generator=False):
    if debug_generator:
        print(f"Judge-question generator request URL: {OLLAMA_CHAT_URL}", file=sys.stderr)
        print("Judge-question generator request JSON:", file=sys.stderr)
        print(json.dumps(payload, indent=2), file=sys.stderr)

    response = post_chat(payload, timeout=300)
    if debug_generator:
        print(f"Judge-question generator HTTP status code: {response.status_code}", file=sys.stderr)
        print("Judge-question generator raw response text:", file=sys.stderr)
        print(response.text, file=sys.stderr)

    response.raise_for_status()
    try:
        response_json = response.json()
    except ValueError as error:
        if debug_generator:
            print(f"Judge-question generator parse error: {error}", file=sys.stderr)
        raise

    if debug_generator:
        print("Judge-question generator parsed response JSON:", file=sys.stderr)
        print(json.dumps(response_json, indent=2), file=sys.stderr)

    raw_text = extract_dynamic_judge_question_text(response_json)
    if debug_generator:
        print("Extracted judge-question generator text repr:", file=sys.stderr)
        print(repr(raw_text), file=sys.stderr)

    try:
        return parse_dynamic_judge_question_json(raw_text)
    except ValueError as error:
        if debug_generator:
            print(f"Judge-question generator parse error: {error}", file=sys.stderr)
        raise


def build_judge_question_payload(generator_model, prompt):
    return {
        "model": generator_model,
        "think": False,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "options": {
            "think": False,
            "temperature": 0.2,
            "top_p": 0.9,
            "num_predict": 700,
            "num_ctx": 4096,
        },
    }


def correct_judge_rubric(
    generator_model,
    use_case,
    use_case_context,
    input_text,
    bad_rubric,
    debug_generator=False,
):
    payload = build_judge_question_payload(
        generator_model,
        build_rubric_correction_prompt(use_case, use_case_context, input_text, bad_rubric),
    )
    parsed_json = request_judge_question_json(generator_model, payload, debug_generator)
    corrected_rubric = parsed_json.get("judge_rubric")
    if not isinstance(corrected_rubric, str) or not corrected_rubric.strip():
        raise ValueError("Generated judge rubric correction is missing judge_rubric.")
    corrected_rubric = corrected_rubric.strip()
    if not is_ordered_judge_rubric(corrected_rubric):
        raise ValueError("Generated judge rubric correction is not ordered.")
    return corrected_rubric


def generate_judge_question_fields(
    generator_model,
    use_case,
    use_case_context,
    test_id,
    input_text,
    debug_generator=False,
):
    payload = build_judge_question_payload(
        generator_model,
        build_dynamic_judge_question_prompt(use_case, use_case_context, test_id, input_text),
    )

    if debug_generator:
        print(f"Judge-question generator test_id: {test_id}", file=sys.stderr)
        print(f"Judge-question generator model: {generator_model}", file=sys.stderr)

    parsed_json = request_judge_question_json(generator_model, payload, debug_generator)
    fields = normalize_generated_judge_fields(parsed_json)
    judge_rubric = fields["judge_rubric"]
    if not is_ordered_judge_rubric(judge_rubric):
        if debug_generator:
            print(
                f"Judge rubric validation failed for {test_id}; attempting correction.",
                file=sys.stderr,
            )
        try:
            judge_rubric = correct_judge_rubric(
                generator_model,
                use_case,
                use_case_context,
                input_text,
                judge_rubric,
                debug_generator,
            )
        except ValueError:
            if debug_generator:
                print(f"Using fallback ordered rubric for {test_id}", file=sys.stderr)
            judge_rubric = build_judge_rubric(use_case)

    return {
        "test_id": test_id,
        "use_case": use_case,
        "use_case_context": use_case_context or "",
        "input": input_text,
        "expected_behavior": fields["expected_behavior"],
        "judge_role": build_judge_role(use_case),
        "judge_standard": build_judge_standard(use_case, use_case_context),
        "judge_rubric": judge_rubric,
        "judge_output_format": build_judge_output_format(use_case),
    }


def build_judge_question_rows(
    tests,
    use_case,
    use_case_context="",
    generator_model=None,
    debug_generator=False,
):
    judge_rows = []
    for test in tests:
        row_use_case = test.get("use_case") or use_case
        row_context = test.get("use_case_context") or use_case_context or ""
        if generator_model:
            try:
                judge_rows.append(
                    generate_judge_question_fields(
                        generator_model,
                        row_use_case,
                        row_context,
                        test["test_id"],
                        test["input"],
                        debug_generator,
                    )
                )
                continue
            except ValueError:
                if debug_generator:
                    print(
                        f"Using fallback judge-question template for {test['test_id']}",
                        file=sys.stderr,
                    )
                    print(f"Using fallback ordered rubric for {test['test_id']}", file=sys.stderr)
        judge_rows.append(
            build_judge_question_fields(
                row_use_case,
                row_context,
                test["test_id"],
                test["input"],
            )
        )
    return judge_rows


def _assert_judge_question_field_templates():
    sample = build_judge_question_fields(
        "sample assistant",
        "",
        "sample_001",
        "Please help with this task.",
    )
    assert "Senior Voice Assistant" not in sample["judge_rubric"]


_assert_judge_question_field_templates()
