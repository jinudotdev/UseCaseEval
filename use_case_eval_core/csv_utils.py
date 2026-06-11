import csv
import io
import re

from .schemas import (
    FINAL_RESULTS_COLUMNS,
    GENERATED_QUESTIONS_COLUMNS,
    JUDGE_1_RESULT_COLUMNS,
    JUDGE_QUESTION_COLUMNS,
    JUDGE_SCORES_COLUMNS,
    JUDGE_TEST_COLUMNS,
    MODEL_RETURNS_COLUMNS,
)


def read_tests(input_path):
    with open(input_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = {"test_id", "input"} - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Input CSV is missing required column(s): {missing}")
        return list(reader)


def write_csv(output_path, fieldnames, rows):
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_generated_tests(output_path, tests):
    write_csv(output_path, GENERATED_QUESTIONS_COLUMNS, tests)


def write_results(output_path, rows):
    write_csv(output_path, FINAL_RESULTS_COLUMNS, rows)


def write_model_returns(output_path, rows):
    write_csv(output_path, MODEL_RETURNS_COLUMNS, rows)


def read_model_returns(input_path):
    with open(input_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = set(MODEL_RETURNS_COLUMNS) - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Model returns CSV is missing required column(s): {missing}")
        return list(reader)


def write_judge_questions(output_path, rows):
    write_csv(output_path, JUDGE_QUESTION_COLUMNS, rows)


def read_judge_questions(input_path):
    with open(input_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = set(JUDGE_QUESTION_COLUMNS) - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Judge questions CSV is missing required column(s): {missing}")
        return list(reader)


def write_judge_scores(output_path, rows):
    write_csv(output_path, JUDGE_SCORES_COLUMNS, rows)


def read_judge_tests(input_path):
    with open(input_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = set(JUDGE_TEST_COLUMNS) - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Judge test CSV is missing required column(s): {missing}")
        return list(reader)


def write_judge_1_results(output_path, rows):
    write_csv(output_path, JUDGE_1_RESULT_COLUMNS, rows)


def slugify_use_case(use_case):
    slug = re.sub(r"[^a-z0-9]+", "_", use_case.strip().lower())
    return slug.strip("_") or "generated_test"


def normalize_generated_header(header):
    normalized = header.strip().lstrip("\ufeff").lower()
    aliases = {
        "id": "test_id",
        "prompt": "input",
    }
    return aliases.get(normalized, normalized)


def strip_markdown_fences(text):
    lines = [
        line
        for line in text.strip().splitlines()
        if not line.strip().startswith("```")
    ]
    return "\n".join(lines).strip()


def clean_generated_csv_text(raw_csv):
    cleaned = strip_markdown_fences(raw_csv)
    lines = cleaned.splitlines()

    for index, line in enumerate(lines):
        try:
            header = next(csv.reader([line], strict=True))
        except csv.Error:
            continue
        normalized_headers = {normalize_generated_header(column) for column in header}
        if {"test_id", "input"}.issubset(normalized_headers):
            return "\n".join(lines[index:]).strip()

    return cleaned


def parse_generated_tests(raw_csv, use_case, num_tests):
    cleaned_csv = clean_generated_csv_text(raw_csv)

    try:
        reader = csv.reader(io.StringIO(cleaned_csv), strict=True)
        rows = list(reader)
    except csv.Error as error:
        raise ValueError(f"Generated CSV could not be parsed: {error}") from error

    if not rows:
        raise ValueError("Generated CSV is empty.")

    normalized_headers = [normalize_generated_header(header) for header in rows[0]]
    missing_columns = {"test_id", "input"} - set(normalized_headers)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Generated CSV is missing required column(s): {missing}")

    test_id_index = normalized_headers.index("test_id")
    input_index = normalized_headers.index("input")
    data_rows = rows[1:]

    if len(data_rows) != num_tests:
        raise ValueError(f"Generated CSV contained {len(data_rows)} prompt(s), expected {num_tests}.")

    prefix = slugify_use_case(use_case)
    tests = []
    for index, row in enumerate(data_rows, start=1):
        if len(row) <= max(test_id_index, input_index):
            raise ValueError(f"Generated CSV row {index} is missing required field(s).")
        if len(row) > len(normalized_headers):
            raise ValueError(f"Generated CSV row {index} has extra unquoted field(s).")
        prompt = row[input_index].strip()
        if not prompt:
            raise ValueError(f"Generated CSV row {index} has an empty input.")
        tests.append(
            {
                "test_id": f"{prefix}_{index:03d}",
                "input": prompt,
            }
        )

    return tests
