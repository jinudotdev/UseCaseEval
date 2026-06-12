import argparse
import sys

from use_case_eval_core.config import (
    DEFAULT_CONFIG_PATH,
    apply_config_defaults,
    load_config,
    validate_model_roles,
)
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
from use_case_eval_core.generate_final_results import build_final_result_rows
from use_case_eval_core.generate_judge_question import build_judge_question_rows
from use_case_eval_core.generate_judge_scores import run_judge_1_batch, run_judge_scores_for_slots
from use_case_eval_core.generate_model_returns import run_eval, run_model_returns
from use_case_eval_core.ollama_client import requests
from use_case_eval_core.generate_question import (
    generate_tests_csv,
    prompt_for_use_case,
)
from use_case_eval_core.schemas import (
    JUDGE_1_RESULTS_OUTPUT,
    JUDGE_TESTS_OUTPUT,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run local Ollama models against a shared CSV of prompts."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"TOML config path. Default: {DEFAULT_CONFIG_PATH}.",
    )
    parser.add_argument("--use-case", help="Name of the use case being evaluated.")
    parser.add_argument(
        "--models",
        help=(
            "Comma-separated Ollama model names. "
            "Default comes from config [models].evaluated."
        ),
    )
    parser.add_argument("--input", help="Input CSV path. Expected columns: test_id,input.")
    parser.add_argument("--output", help="Output CSV path for model responses.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate per response. Default comes from config [run].max_tokens.",
    )
    parser.add_argument(
        "--generate-tests",
        action="store_true",
        help="Generate generated_questions.csv with a local Ollama model.",
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=None,
        help="Number of questions to generate. Default comes from config [run].num_tests.",
    )
    parser.add_argument(
        "--generator-model",
        default=None,
        help="Ollama model to use for test generation. Default comes from config [models].frontier.",
    )
    parser.add_argument(
        "--generated-questions",
        dest="generated_questions",
        default=None,
        help="CSV path for generated questions. Default comes from config [paths].generated_questions.",
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
        default=None,
        help="Model returns output CSV path. Default comes from config [paths].generated_model_returns.",
    )
    parser.add_argument(
        "--generate-judge-scores",
        action="store_true",
        help="Score generated_model_returns.csv using generated_judge_questions.csv.",
    )
    parser.add_argument(
        "--judge-questions-input",
        default=None,
        help="Judge questions input CSV path. Default comes from config [paths].generated_judge_questions.",
    )
    parser.add_argument(
        "--judge-questions-output",
        default=None,
        help="Judge questions output CSV path. Default comes from config [paths].generated_judge_questions.",
    )
    parser.add_argument(
        "--model-returns-input",
        default=None,
        help="Model returns input CSV path. Default comes from config [paths].generated_model_returns.",
    )
    parser.add_argument(
        "--judge-scores-output",
        default=None,
        help="Judge scores output CSV path. Default comes from config [paths].generated_judge_scores.",
    )
    parser.add_argument(
        "--judge-scores-input",
        default=None,
        help="Judge scores input CSV path. Default comes from config [paths].generated_judge_scores.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Ollama model to use for generated judge scoring. Default comes from config [models].judge_1.",
    )
    parser.add_argument(
        "--judge-2-model",
        default=None,
        help="Optional Ollama model to use for Judge 2. Default comes from config [models].judge_2.",
    )
    parser.add_argument(
        "--judge-pass-threshold",
        type=int,
        default=None,
        help="Minimum generated judge score required to pass. Default comes from config [run].judge_pass_threshold.",
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
        default=None,
        help="Final results output CSV path. Default comes from config [paths].final_results.",
    )
    parser.add_argument(
        "--judge-1-model",
        help="Ollama model to use as Judge 1. Required with --run-judge-1.",
    )
    parser.add_argument(
        "--judge-1-threshold",
        type=int,
        default=None,
        help="Minimum Judge 1 score required to pass. Default comes from config [run].judge_pass_threshold.",
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
    return parse_model_names(args.models)


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
        print(
            f"No new generated questions were written; any previous {args.generated_questions} was not overwritten.",
            file=sys.stderr,
        )
        raise

    write_generated_tests(args.generated_questions, tests)
    print(f"Wrote {len(tests)} generated questions to {args.generated_questions}")
    return tests


def generate_judge_questions_file(
    tests,
    use_case,
    generator_model,
    output_path,
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
    write_judge_questions(output_path, judge_question_rows)
    print(f"Wrote {len(judge_question_rows)} judge question rows to {output_path}")
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
    judge_models,
):
    judge_score_rows = run_judge_scores_for_slots(
        judge_questions,
        model_returns,
        judge_models,
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
        validate_model_roles(required_models=[("frontier", args.generator_model)])
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
        validate_model_roles(
            required_models=[("frontier", args.judge_question_generator_model)]
        )
        tests = read_tests(input_path)
        generate_judge_questions_file(
            tests,
            resolve_use_case(args),
            args.judge_question_generator_model,
            args.judge_questions_output,
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
        validation = validate_model_roles(
            evaluated_models=resolve_model_names(args),
            require_evaluated=True,
        )
        tests = read_tests(args.input or args.generated_questions)
        rows = generate_model_returns_file(
            args,
            resolve_use_case(args),
            validation["evaluated"],
            tests,
        )
    except (OSError, ValueError) as error:
        print(f"Error generating model returns: {error}", file=sys.stderr)
        return 1
    return 0 if rows is not None else 1


def run_generate_judge_scores_workflow(args):
    judge_model = args.judge_model
    try:
        validation = validate_model_roles(
            required_models=[("judge_1", judge_model)],
            optional_models=[("judge_2", getattr(args, "judge_2_model", ""))],
        )
        judge_models = [("judge_1", judge_model)]
        judge_2_model = validation["optional"].get("judge_2", "")
        if judge_2_model:
            judge_models.append(("judge_2", judge_2_model))
        judge_questions = read_judge_questions(args.judge_questions_input)
        model_returns = read_model_returns(args.model_returns_input)
        generate_judge_scores_file(args, judge_questions, model_returns, judge_models)
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
    judge_question_generator_model = args.judge_question_generator_model
    judge_model = args.judge_model

    try:
        validate_positive(args.num_tests, "--num-tests")
        validate_positive(args.max_tokens, "--max-tokens")
        validation = validate_model_roles(
            required_models=[
                ("frontier", args.generator_model),
                ("judge-question generator", judge_question_generator_model),
                ("judge_1", judge_model),
            ],
            optional_models=[("judge_2", getattr(args, "judge_2_model", ""))],
            evaluated_models=model_names,
            require_evaluated=True,
        )
        model_names = validation["evaluated"]
        args.judge_2_model = validation["optional"].get("judge_2", "")
        judge_models = [("judge_1", judge_model)]
        if args.judge_2_model:
            judge_models.append(("judge_2", args.judge_2_model))

        print("Starting full UseCaseEval pipeline.")
        print(f"Use case: {use_case}")
        print(f"Tested models: {','.join(model_names)}")

        print(f"Step 1/5: Generate {args.generated_questions}")
        generate_questions_file(args, use_case)

        print(f"Step 2/5: Generate {args.judge_questions_output}")
        tests = read_tests(args.generated_questions)
        generate_judge_questions_file(
            tests,
            use_case,
            judge_question_generator_model,
            args.judge_questions_output,
            args.debug_judge_question_generator,
        )

        print(f"Step 3/5: Generate {args.model_returns_output}")
        tests = read_tests(args.generated_questions)
        model_return_rows = generate_model_returns_file(args, use_case, model_names, tests)
        if model_return_rows is None:
            return 1

        print(f"Step 4/5: Generate {args.judge_scores_output}")
        judge_questions = read_judge_questions(args.judge_questions_output)
        model_returns = read_model_returns(args.model_returns_output)
        generate_judge_scores_file(args, judge_questions, model_returns, judge_models)

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

    try:
        config = load_config(args.config)
        apply_config_defaults(args, config)
    except (OSError, ValueError) as error:
        print(f"Error loading config: {error}", file=sys.stderr)
        return 1

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
            validate_model_roles(required_models=[("judge_1", judge_1_model)])
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
