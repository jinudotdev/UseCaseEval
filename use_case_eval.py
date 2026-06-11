import argparse
import sys

from use_case_eval_core.csv_utils import (
    parse_generated_tests,
    read_judge_tests,
    read_tests,
    write_generated_tests,
    write_judge_1_results,
    write_judge_questions,
    write_model_returns,
    write_results,
)
from use_case_eval_core.judge_question_generation import build_judge_question_rows
from use_case_eval_core.judge_runner import run_judge_1_batch
from use_case_eval_core.model_runner import run_eval, run_model_returns
from use_case_eval_core.ollama_client import requests
from use_case_eval_core.question_generation import (
    generate_tests_csv,
    prompt_for_num_tests,
    prompt_for_use_case,
)
from use_case_eval_core.schemas import (
    GENERATED_QUESTIONS_OUTPUT,
    JUDGE_1_RESULTS_OUTPUT,
    JUDGE_QUESTIONS_OUTPUT,
    JUDGE_TESTS_OUTPUT,
    MODEL_RETURNS_OUTPUT,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run local Ollama models against a shared CSV of prompts."
    )
    parser.add_argument("--use-case", help="Name of the use case being evaluated.")
    parser.add_argument(
        "--models",
        help="Comma-separated Ollama model names, for example qwen35-9b,llama32-1b-q4.",
    )
    parser.add_argument("--input", help="Input CSV path. Expected columns: test_id,input.")
    parser.add_argument("--output", help="Output CSV path for model responses.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=80,
        help="Maximum number of tokens to generate per response. Default: 80.",
    )
    parser.add_argument(
        "--generate-tests",
        action="store_true",
        help="Interactively generate questions with a local Ollama model before running eval.",
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=None,
        help="Number of questions to generate. Default: 10.",
    )
    parser.add_argument(
        "--generator-model",
        default="qwen35-9b",
        help="Ollama model to use for test generation. Default: qwen35-9b.",
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
        help="Run tested models and write raw generated_model_returns.csv responses.",
    )
    parser.add_argument(
        "--model-returns-output",
        default=MODEL_RETURNS_OUTPUT,
        help=f"Model returns output CSV path. Default: {MODEL_RETURNS_OUTPUT}.",
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


def main():
    args = parse_args()
    judge_1_model = args.judge_1_model.strip() if args.judge_1_model else None

    if requests is None:
        print(
            "Error: The requests package is not installed. "
            "Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

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

    if args.export_judge_questions:
        if not args.use_case:
            print("Error: --use-case is required with --export-judge-questions.", file=sys.stderr)
            return 1
        if not args.input:
            print("Error: --input is required with --export-judge-questions.", file=sys.stderr)
            return 1
        try:
            tests = read_tests(args.input)
            judge_question_rows = build_judge_question_rows(
                tests,
                args.use_case,
                args.judge_question_generator_model,
                args.debug_judge_question_generator,
            )
            write_judge_questions(JUDGE_QUESTIONS_OUTPUT, judge_question_rows)
        except (OSError, ValueError) as error:
            print(f"Error writing judge questions CSV: {error}", file=sys.stderr)
            return 1

        print(f"Wrote {len(judge_question_rows)} judge question rows to {JUDGE_QUESTIONS_OUTPUT}")
        return 0

    model_names = [model.strip() for model in (args.models or "").split(",") if model.strip()]

    if args.generate_model_returns:
        if not args.use_case:
            print("Error: --use-case is required with --generate-model-returns.", file=sys.stderr)
            return 1
        if not args.input:
            print("Error: --input is required with --generate-model-returns.", file=sys.stderr)
            return 1
        if not model_names:
            print("Error: --models must include at least one model name.", file=sys.stderr)
            return 1
        if args.max_tokens < 1:
            print("Error: --max-tokens must be 1 or greater.", file=sys.stderr)
            return 1
        try:
            tests = read_tests(args.input)
        except (OSError, ValueError) as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1

        rows = run_model_returns(args.use_case, model_names, tests, args.max_tokens)
        if rows is None:
            return 1

        try:
            write_model_returns(args.model_returns_output, rows)
        except OSError as error:
            print(f"Error writing model returns CSV: {error}", file=sys.stderr)
            return 1

        print(f"Wrote {len(rows)} model return rows to {args.model_returns_output}")
        return 0

    if not model_names:
        print("Error: --models must include at least one model name.", file=sys.stderr)
        return 1

    if not args.output:
        print("Error: --output is required unless --run-judge-1 is used.", file=sys.stderr)
        return 1

    if args.max_tokens < 1:
        print("Error: --max-tokens must be 1 or greater.", file=sys.stderr)
        return 1

    if args.generate_tests:
        use_case = prompt_for_use_case()
        num_tests = args.num_tests if args.num_tests is not None else prompt_for_num_tests()
        if num_tests < 1:
            print("Error: --num-tests must be 1 or greater.", file=sys.stderr)
            return 1

        print(f"Generating {num_tests} question(s) with {args.generator_model}...")
        try:
            raw_generated_csv = generate_tests_csv(
                args.generator_model,
                use_case,
                num_tests,
                args.debug_generator,
            )
            tests = parse_generated_tests(raw_generated_csv, use_case, num_tests)
            write_generated_tests(args.generated_questions, tests)
        except requests.exceptions.ConnectionError:
            print(
                "Error: Could not connect to Ollama at http://localhost:11434. "
                "Make sure Ollama is running, then try again.",
                file=sys.stderr,
            )
            return 1
        except requests.exceptions.RequestException as error:
            print(f"Error generating questions with Ollama: {error}", file=sys.stderr)
            return 1
        except (OSError, ValueError) as error:
            if "raw_generated_csv" in locals():
                print("Raw generator output:", file=sys.stderr)
                print(repr(raw_generated_csv), file=sys.stderr)
            print(f"Error generating questions: {error}", file=sys.stderr)
            return 1

        print(f"Wrote {len(tests)} generated questions to {args.generated_questions}")
    else:
        if not args.use_case:
            print("Error: --use-case is required unless --generate-tests is used.", file=sys.stderr)
            return 1
        if not args.input:
            print("Error: --input is required unless --generate-tests is used.", file=sys.stderr)
            return 1
        use_case = args.use_case
        try:
            tests = read_tests(args.input)
        except (OSError, ValueError) as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1

    rows = run_eval(
        use_case,
        model_names,
        tests,
        args.max_tokens,
    )
    if rows is None:
        return 1

    try:
        write_results(args.output, rows)
    except OSError as error:
        print(f"Error writing output CSV: {error}", file=sys.stderr)
        return 1

    print(f"Wrote {len(rows)} result rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
