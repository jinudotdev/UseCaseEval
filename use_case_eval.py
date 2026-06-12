import argparse
import sys

from use_case_eval_core.csv_utils import (
    parse_generated_tests,
    read_judge_questions,
    read_judge_scores,
    read_judge_tests,
    read_model_returns,
    read_tests,
    write_final_results,
    write_generated_tests,
    write_judge_1_results,
    write_judge_questions,
    write_judge_scores,
    write_model_returns,
    write_results,
)
from use_case_eval_core.final_results_generation import build_final_result_rows
from use_case_eval_core.judge_question_generation import build_judge_question_rows
from use_case_eval_core.judge_scores_generation import run_judge_1_batch, run_judge_scores
from use_case_eval_core.model_returns_generation import run_eval, run_model_returns
from use_case_eval_core.ollama_client import requests
from use_case_eval_core.question_generation import (
    generate_tests_csv,
    prompt_for_use_case,
)
from use_case_eval_core.schemas import (
    FINAL_RESULTS_OUTPUT,
    GENERATED_QUESTIONS_OUTPUT,
    JUDGE_1_RESULTS_OUTPUT,
    JUDGE_QUESTIONS_OUTPUT,
    JUDGE_SCORES_OUTPUT,
    JUDGE_TESTS_OUTPUT,
    MODEL_RETURNS_OUTPUT,
)


DEFAULT_NUM_TESTS = 10
DEFAULT_GENERATOR_MODEL = "qwen35-9b"
DEFAULT_JUDGE_QUESTION_GENERATOR_MODEL = "qwen35-9b"
DEFAULT_JUDGE_MODEL = "qwen35-9b"
DEFAULT_TESTED_MODELS = [
    "qwen25-15b-q4",
    "llama32-1b-q4",
    "tinyllama-11b-q4",
    "smollm2-17b-q4",
    "qwen25-05b-q8",
]
DEFAULT_TESTED_MODELS_CSV = ",".join(DEFAULT_TESTED_MODELS)
DEFAULT_MAX_TOKENS = 220


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run local Ollama models against a shared CSV of prompts."
    )
    parser.add_argument("--use-case", help="Name of the use case being evaluated.")
    parser.add_argument(
        "--models",
        help=(
            "Comma-separated Ollama model names. "
            f"Default for full/model-return workflows: {DEFAULT_TESTED_MODELS_CSV}."
        ),
    )
    parser.add_argument("--input", help="Input CSV path. Expected columns: test_id,input.")
    parser.add_argument("--output", help="Output CSV path for model responses.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum number of tokens to generate per response. Default: {DEFAULT_MAX_TOKENS}.",
    )
    parser.add_argument(
        "--generate-tests",
        action="store_true",
        help="Generate generated_questions.csv with a local Ollama model.",
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=DEFAULT_NUM_TESTS,
        help=f"Number of questions to generate. Default: {DEFAULT_NUM_TESTS}.",
    )
    parser.add_argument(
        "--generator-model",
        default=DEFAULT_GENERATOR_MODEL,
        help=f"Ollama model to use for test generation. Default: {DEFAULT_GENERATOR_MODEL}.",
    )
    parser.add_argument(
        "--generated-questions",
        dest="generated_questions",
        default=GENERATED_QUESTIONS_OUTPUT,
        help=f"CSV path for generated questions. Default: {GENERATED_QUESTIONS_OUTPUT}.",
    )
    parser.add_argument(
        "--generated-input",
        dest="generated_questions",
        default=argparse.SUPPRESS,
        help="Deprecated alias for --generated-questions.",
    )
    parser.add_argument(
        "--debug-generator",
        action="store_true",
        help="Print Ollama generator request and response diagnostics.",
    )
    parser.add_argument(
        "--generate-model-returns",
        action="store_true",
        help="Run tested models against generated_questions.csv and write raw responses.",
    )
    parser.add_argument(
        "--model-returns-output",
        default=MODEL_RETURNS_OUTPUT,
        help=f"Model returns output CSV path. Default: {MODEL_RETURNS_OUTPUT}.",
    )
    parser.add_argument(
        "--generate-judge-scores",
        action="store_true",
        help="Score generated_model_returns.csv using generated_judge_questions.csv.",
    )
    parser.add_argument(
        "--judge-questions-input",
        default=".\\generated_judge_questions.csv",
        help="Judge questions input CSV path. Default: .\\generated_judge_questions.csv.",
    )
    parser.add_argument(
        "--model-returns-input",
        default=MODEL_RETURNS_OUTPUT,
        help=f"Model returns input CSV path. Default: {MODEL_RETURNS_OUTPUT}.",
    )
    parser.add_argument(
        "--judge-scores-output",
        default=JUDGE_SCORES_OUTPUT,
        help=f"Judge scores output CSV path. Default: {JUDGE_SCORES_OUTPUT}.",
    )
    parser.add_argument(
        "--judge-scores-input",
        default=JUDGE_SCORES_OUTPUT,
        help=f"Judge scores input CSV path. Default: {JUDGE_SCORES_OUTPUT}.",
    )
    parser.add_argument(
        "--judge-model",
        help=f"Ollama model to use for generated judge scoring. Default: {DEFAULT_JUDGE_MODEL}.",
    )
    parser.add_argument(
        "--judge-pass-threshold",
        type=int,
        default=4,
        help="Minimum generated judge score required to pass. Default: 4.",
    )
    parser.add_argument(
        "--debug-judge-scores",
        action="store_true",
        help="Print Ollama generated judge score request and response diagnostics.",
    )
    parser.add_argument(
        "--generate-final-results",
        action="store_true",
        help="Merge generated_model_returns.csv and generated_judge_scores.csv into final_results.csv.",
    )
    parser.add_argument(
        "--final-results-output",
        default=FINAL_RESULTS_OUTPUT,
        help=f"Final results output CSV path. Default: {FINAL_RESULTS_OUTPUT}.",
    )
    parser.add_argument(
        "--judge-1-model",
        help="Ollama model to use as Judge 1. Required with --run-judge-1.",
    )
    parser.add_argument(
        "--judge-1-threshold",
        type=int,
        default=4,
        help="Minimum Judge 1 score required to pass. Default: 4.",
    )
    parser.add_argument(
        "--debug-judge-1",
        action="store_true",
        help="Print Ollama Judge 1 request and response diagnostics.",
    )
    parser.add_argument(
        "--export-judge-questions",
        "--export-judge-tests",
        dest="export_judge_questions",
        action="store_true",
        help=(
            "Write generated_judge_questions.csv with judge instructions for each "
            "question. --export-judge-tests is a deprecated alias."
        ),
    )
    parser.add_argument(
        "--judge-question-generator-model",
        help="Optional Ollama model to dynamically generate judge question instructions.",
    )
    parser.add_argument(
        "--debug-judge-question-generator",
        action="store_true",
        help="Print Ollama judge-question generator diagnostics.",
    )
    parser.add_argument(
        "--run-judge-1",
        action="store_true",
        help="Read generated_judge_tests.csv and write Judge 1 scores.",
    )
    parser.add_argument(
        "--judge-1-input",
        default=JUDGE_TESTS_OUTPUT,
        help=f"Judge 1 input CSV path. Default: {JUDGE_TESTS_OUTPUT}.",
    )
    parser.add_argument(
        "--judge-1-output",
        default=JUDGE_1_RESULTS_OUTPUT,
        help=f"Judge 1 output CSV path. Default: {JUDGE_1_RESULTS_OUTPUT}.",
    )
    return parser.parse_args()


def parse_model_names(models):
    return [model.strip() for model in (models or "").split(",") if model.strip()]


def resolve_model_names(args):
    return parse_model_names(args.models) or list(DEFAULT_TESTED_MODELS)


def has_workflow_flag(args):
    return any(
        [
            args.generate_tests,
            args.export_judge_questions,
            args.generate_model_returns,
            args.generate_judge_scores,
            args.generate_final_results,
            args.run_judge_1,
        ]
    )


def resolve_use_case(args):
    if args.use_case and args.use_case.strip():
        return args.use_case.strip()
    return prompt_for_use_case()


def validate_positive(value, option_name):
    if value < 1:
        raise ValueError(f"{option_name} must be 1 or greater.")


def print_ollama_connection_error():
    print(
        "Error: Could not connect to Ollama at http://localhost:11434. "
        "Make sure Ollama is running, then try again.",
        file=sys.stderr,
    )


def generate_questions_file(args, use_case):
    validate_positive(args.num_tests, "--num-tests")
    print(f"Generating {args.num_tests} question(s) with {args.generator_model}...")

    raw_generated_csv = generate_tests_csv(
        args.generator_model,
        use_case,
        args.num_tests,
        args.debug_generator,
    )
    try:
        tests = parse_generated_tests(raw_generated_csv, use_case, args.num_tests)
    except ValueError:
        print("Raw generator output:", file=sys.stderr)
        print(repr(raw_generated_csv), file=sys.stderr)
        raise

    write_generated_tests(args.generated_questions, tests)
    print(f"Wrote {len(tests)} generated questions to {args.generated_questions}")
    return tests


def generate_judge_questions_file(
    tests,
    use_case,
    generator_model,
    debug_judge_question_generator=False,
):
    generator_label = generator_model or "fallback templates"
    print(f"Generating judge questions with {generator_label}...")
    judge_question_rows = build_judge_question_rows(
        tests,
        use_case,
        generator_model,
        debug_judge_question_generator,
    )
    write_judge_questions(JUDGE_QUESTIONS_OUTPUT, judge_question_rows)
    print(f"Wrote {len(judge_question_rows)} judge question rows to {JUDGE_QUESTIONS_OUTPUT}")
    return judge_question_rows


def generate_model_returns_file(args, use_case, model_names, tests):
    validate_positive(args.max_tokens, "--max-tokens")
    rows = run_model_returns(use_case, model_names, tests, args.max_tokens)
    if rows is None:
        return None
    write_model_returns(args.model_returns_output, rows)
    print(f"Wrote {len(rows)} model return rows to {args.model_returns_output}")
    return rows


def generate_judge_scores_file(
    args,
    judge_questions,
    model_returns,
    judge_model,
):
    judge_score_rows = run_judge_scores(
        judge_questions,
        model_returns,
        judge_model,
        args.judge_pass_threshold,
        args.debug_judge_scores,
    )
    write_judge_scores(args.judge_scores_output, judge_score_rows)
    print(f"Wrote {len(judge_score_rows)} judge score rows to {args.judge_scores_output}")
    return judge_score_rows


def generate_final_results_file(args, model_returns, judge_scores):
    final_result_rows = build_final_result_rows(model_returns, judge_scores)
    write_final_results(args.final_results_output, final_result_rows)
    print(f"Wrote {len(final_result_rows)} final result rows to {args.final_results_output}")
    return final_result_rows


def run_generate_tests_workflow(args):
    try:
        generate_questions_file(args, resolve_use_case(args))
    except requests.exceptions.ConnectionError:
        print_ollama_connection_error()
        return 1
    except requests.exceptions.RequestException as error:
        print(f"Error generating questions with Ollama: {error}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"Error generating questions: {error}", file=sys.stderr)
        return 1
    return 0


def run_export_judge_questions_workflow(args):
    input_path = args.input or args.generated_questions
    try:
        tests = read_tests(input_path)
        generate_judge_questions_file(
            tests,
            resolve_use_case(args),
            args.judge_question_generator_model,
            args.debug_judge_question_generator,
        )
    except requests.exceptions.ConnectionError:
        print_ollama_connection_error()
        return 1
    except requests.exceptions.RequestException as error:
        print(f"Error generating judge questions with Ollama: {error}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"Error writing judge questions CSV: {error}", file=sys.stderr)
        return 1
    return 0


def run_generate_model_returns_workflow(args):
    try:
        tests = read_tests(args.input or args.generated_questions)
        rows = generate_model_returns_file(args, resolve_use_case(args), resolve_model_names(args), tests)
    except (OSError, ValueError) as error:
        print(f"Error generating model returns: {error}", file=sys.stderr)
        return 1
    return 0 if rows is not None else 1


def run_generate_judge_scores_workflow(args):
    judge_model = args.judge_model or DEFAULT_JUDGE_MODEL
    try:
        judge_questions = read_judge_questions(args.judge_questions_input)
        model_returns = read_model_returns(args.model_returns_input)
        generate_judge_scores_file(args, judge_questions, model_returns, judge_model)
    except (OSError, ValueError) as error:
        print(f"Error generating judge scores: {error}", file=sys.stderr)
        return 1
    return 0


def run_generate_final_results_workflow(args):
    try:
        model_returns = read_model_returns(args.model_returns_input)
        judge_scores = read_judge_scores(args.judge_scores_input)
        generate_final_results_file(args, model_returns, judge_scores)
    except (OSError, ValueError) as error:
        print(f"Error generating final results: {error}", file=sys.stderr)
        return 1
    return 0


def run_full_pipeline(args):
    use_case = resolve_use_case(args)
    model_names = resolve_model_names(args)
    judge_question_generator_model = (
        args.judge_question_generator_model or DEFAULT_JUDGE_QUESTION_GENERATOR_MODEL
    )
    judge_model = args.judge_model or DEFAULT_JUDGE_MODEL

    try:
        validate_positive(args.num_tests, "--num-tests")
        validate_positive(args.max_tokens, "--max-tokens")

        print("Starting full UseCaseEval pipeline.")
        print(f"Use case: {use_case}")
        print(f"Tested models: {','.join(model_names)}")

        print(f"Step 1/5: Generate {args.generated_questions}")
        generate_questions_file(args, use_case)

        print(f"Step 2/5: Generate {JUDGE_QUESTIONS_OUTPUT}")
        tests = read_tests(args.generated_questions)
        generate_judge_questions_file(
            tests,
            use_case,
            judge_question_generator_model,
            args.debug_judge_question_generator,
        )

        print(f"Step 3/5: Generate {args.model_returns_output}")
        tests = read_tests(args.generated_questions)
        model_return_rows = generate_model_returns_file(args, use_case, model_names, tests)
        if model_return_rows is None:
            return 1

        print(f"Step 4/5: Generate {args.judge_scores_output}")
        judge_questions = read_judge_questions(JUDGE_QUESTIONS_OUTPUT)
        model_returns = read_model_returns(args.model_returns_output)
        generate_judge_scores_file(args, judge_questions, model_returns, judge_model)

        print(f"Step 5/5: Generate {args.final_results_output}")
        model_returns = read_model_returns(args.model_returns_output)
        judge_scores = read_judge_scores(args.judge_scores_output)
        generate_final_results_file(args, model_returns, judge_scores)
    except requests.exceptions.ConnectionError:
        print_ollama_connection_error()
        return 1
    except requests.exceptions.RequestException as error:
        print(f"Error running full pipeline with Ollama: {error}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"Error running full pipeline: {error}", file=sys.stderr)
        return 1

    print("Full pipeline complete.")
    return 0


def run_legacy_eval_workflow(args):
    model_names = parse_model_names(args.models)
    if not model_names:
        print("Error: --models must include at least one model name.", file=sys.stderr)
        return 1

    if not args.output:
        print("Error: --output is required for the legacy model response workflow.", file=sys.stderr)
        return 1

    if args.max_tokens < 1:
        print("Error: --max-tokens must be 1 or greater.", file=sys.stderr)
        return 1

    if not args.use_case:
        print("Error: --use-case is required for the legacy model response workflow.", file=sys.stderr)
        return 1
    if not args.input:
        print("Error: --input is required for the legacy model response workflow.", file=sys.stderr)
        return 1

    try:
        tests = read_tests(args.input)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    rows = run_eval(args.use_case, model_names, tests, args.max_tokens)
    if rows is None:
        return 1

    try:
        write_results(args.output, rows)
    except OSError as error:
        print(f"Error writing output CSV: {error}", file=sys.stderr)
        return 1

    print(f"Wrote {len(rows)} result rows to {args.output}")
    return 0


def main():
    args = parse_args()
    judge_1_model = args.judge_1_model.strip() if args.judge_1_model else None

    if args.generate_final_results:
        return run_generate_final_results_workflow(args)

    if requests is None:
        print(
            "Error: The requests package is not installed. "
            "Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    workflow_requested = has_workflow_flag(args)

    if args.run_judge_1:
        if not judge_1_model:
            print("Error: --judge-1-model is required with --run-judge-1.", file=sys.stderr)
            return 1
        try:
            judge_tasks = read_judge_tests(args.judge_1_input)
            judge_1_rows = run_judge_1_batch(
                judge_tasks,
                judge_1_model,
                args.judge_1_threshold,
                args.debug_judge_1,
            )
            write_judge_1_results(args.judge_1_output, judge_1_rows)
        except (OSError, ValueError) as error:
            print(f"Error running Judge 1: {error}", file=sys.stderr)
            return 1

        print(f"Wrote {len(judge_1_rows)} Judge 1 rows to {args.judge_1_output}")
        return 0

    if args.generate_judge_scores:
        return run_generate_judge_scores_workflow(args)

    if args.export_judge_questions:
        return run_export_judge_questions_workflow(args)

    if args.generate_model_returns:
        return run_generate_model_returns_workflow(args)

    if args.generate_tests:
        return run_generate_tests_workflow(args)

    if args.output and not workflow_requested:
        return run_legacy_eval_workflow(args)

    if not workflow_requested:
        return run_full_pipeline(args)

    print("Error: No workflow selected.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
