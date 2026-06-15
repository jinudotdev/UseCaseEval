import json
import sys

from .ollama_client import extract_generated_text, post_chat
from .schemas import OLLAMA_CHAT_URL


def build_generator_prompt(use_case, use_case_context, num_tests):
    normalized_context = (use_case_context or "").strip()
    context_block = normalized_context or "No additional context supplied."
    return (
        "You are creating an evaluation set for local LLM models.\n"
        "Use case name:\n"
        f"{use_case}\n\n"
        "Use case context:\n"
        f"{context_block}\n\n"
        f"Create exactly {num_tests} test prompts.\n"
        f"Return exactly {num_tests + 1} CSV lines total: one header line plus {num_tests} data rows.\n"
        "If use-case context is supplied, treat it as the authoritative description of the intended assistant.\n"
        "Every generated question must directly match both the use-case name and the supplied context.\n"
        "Do not introduce capabilities, interfaces, audiences, tools, or professional authority that contradict the supplied context.\n"
        "If no context is supplied, use the voice-assistant, offline, self-contained, and safety defaults below.\n"
        "Each prompt should test one realistic spoken task for that use case.\n"
        "The evaluated model is a small, local, conversational voice assistant.\n"
        "Prefer requests that a user would naturally say aloud in one or two sentences.\n"
        "Generated tasks should produce responses that remain useful when spoken aloud.\n"
        "Prefer short, direct, conversational answers instead of long documents, tables, forms, or heavily formatted output.\n"
        "Keep expected responses reasonably brief unless the task clearly requires more detail.\n"
        "When choosing between possible tasks, prefer the one that would be useful in a hands-free voice conversation.\n"
        "Treat the model as a limited general-purpose assistant, not as a licensed, regulated, or highly specialized professional.\n"
        "Do not generate tasks that require the model to act as a doctor, nurse, lawyer, financial adviser, therapist, emergency dispatcher, or another licensed or regulated professional.\n"
        "Do not ask the model to diagnose medical conditions, prescribe or recommend treatment, make legal decisions, give personalized investment instructions, make high-stakes professional judgments, claim authority it does not have, or perform actions through tools or services that are not available.\n"
        "General educational information and escalation guidance are allowed.\n"
        "Prefer practical, bounded voice-assistant tasks that smaller local models can reasonably handle, such as answering factual questions from stable general knowledge or supplied information, brief plain-language explanations, short spoken plans, reminders, step-by-step instructions, interpreting or reorganizing a few routine tasks, rewriting a short message in a calmer or clearer tone, drafting a concise email or text message, summarizing short supplied information, extracting a date, time, name, task, or key detail from supplied text, organizing a few spoken notes into a simple verbal plan, helping the user decide what to do next in an ordinary low-risk situation, asking for missing information when a request cannot be completed accurately, identifying when a user should contact a qualified professional, caregiver, or emergency service, and conversational support that does not claim professional expertise.\n"
        "Do not generate tasks whose usefulness primarily depends on visual formatting, including Markdown checkboxes, complex tables, spreadsheets, multi-column layouts, detailed forms, long written reports, or visually organized weekly planners.\n"
        "Do not ask the assistant to create a checklist unless the result can be expressed naturally as a short spoken sequence.\n"
        "Drafting short text is allowed when it is a realistic voice-assistant task, such as a short email, text message, reminder, calendar title, or short note.\n"
        "For factual questions, include needed facts directly or use stable general knowledge; do not require internet access, live data, private account access, or external tools.\n"
        "When a use case is ambiguous, interpret it as a limited assistant role rather than assuming professional authority.\n"
        "Every input must be self-contained and include all context needed to answer.\n"
        "Do not create inputs that rely on missing prior conversation, hidden context, uploads, images, highlighting, private records, web access, live information, or external tools.\n"
        "Do not create inputs that assume the assistant can access calendars, reminders, bank accounts, medical records, email, contacts, private files, or other services unless the prompt explicitly includes a simulated tool result or supplied data.\n"
        "The assistant may explain general educational information, but it must not present itself as a professional or make high-stakes decisions for the user.\n"
        "The assistant may help draft or describe an action, but it must not falsely claim to have performed an unavailable action.\n"
        "Prompts should be concise, practical, conversational, and varied, but complete enough to answer without extra context.\n"
        "For reading comprehension tutor use cases, include a sentence or short passage inside each input.\n"
        "If asking about a highlighted word, represent it textually, for example: In the sentence \"The tiny mouse was brave,\" the highlighted word is \"tiny\". What is the opposite of the highlighted word?\n"
        "If asking for a summary, include the passage to summarize.\n"
        "Do not include answers.\n"
        "Do not use markdown.\n"
        "Do not wrap the CSV in code fences.\n"
        "Do not add commentary before or after the CSV.\n"
        "Use valid CSV quoting for any input that contains commas or quotation marks.\n"
        "The first line must be exactly: test_id,input\n"
        "Return only valid CSV with columns:\n"
        "test_id,input"
    )


def generate_tests_csv(
    generator_model,
    use_case,
    use_case_context,
    num_tests,
    debug_generator=False,
):
    payload = {
        "model": generator_model,
        "think": False,
        "messages": [
            {
                "role": "user",
                "content": "/no_think\n"
                + build_generator_prompt(use_case, use_case_context, num_tests),
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
        use_case = input("What use case would you like to evaluate? ").strip()
        if use_case:
            return use_case
        print("Please enter a use case.")


def prompt_for_use_case_context():
    print("Briefly describe the intended assistant in 1-4 sentences.")
    print("Include its target users, expected tasks, interface, and important limitations.")
    return input("Press Enter to continue without additional context: ").strip()


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
