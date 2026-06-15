import csv
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
        rows = list(reader)
        add_missing_context_columns(rows)
        return rows


def add_missing_context_columns(rows):
    for row in rows:
        row.setdefault("use_case", "")
        row.setdefault("use_case_context", "")


def write_csv(output_path, fieldnames, rows):
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_generated_tests(output_path, tests):
    write_csv(output_path, GENERATED_QUESTIONS_COLUMNS, tests)


def write_results(output_path, rows):
    write_csv(output_path, FINAL_RESULTS_COLUMNS, rows)


def write_final_results(output_path, rows):
    write_results(output_path, rows)


def write_model_returns(output_path, rows):
    write_csv(output_path, MODEL_RETURNS_COLUMNS, rows)


def read_model_returns(input_path):
    with open(input_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = set(MODEL_RETURNS_COLUMNS) - {"use_case_context"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Model returns CSV is missing required column(s): {missing}")
        rows = list(reader)
        add_missing_context_columns(rows)
        return rows


def write_judge_questions(output_path, rows):
    write_csv(output_path, JUDGE_QUESTION_COLUMNS, rows)


def read_judge_questions(input_path):
    with open(input_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = set(JUDGE_QUESTION_COLUMNS) - {"use_case", "use_case_context"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Judge questions CSV is missing required column(s): {missing}")
        rows = list(reader)
        add_missing_context_columns(rows)
        return rows


def write_judge_scores(output_path, rows):
    write_csv(output_path, JUDGE_SCORES_COLUMNS, rows)


def read_judge_scores(input_path):
    with open(input_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = set(JUDGE_SCORES_COLUMNS) - {"judge_slot", "use_case_context"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Judge scores CSV is missing required column(s): {missing}")
        rows = list(reader)
        add_missing_context_columns(rows)
        for row in rows:
            if not row.get("judge_slot"):
                row["judge_slot"] = "judge_1"
        return rows


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


def split_generated_line(line, row_label):
    if "," not in line:
        raise ValueError(f"Generated CSV {row_label} is missing the test_id/input separator comma.")
    first_field, input_text = line.split(",", 1)
    return first_field.strip(), strip_wrapping_quotes(input_text.strip())


def strip_wrapping_quotes(value):
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        quote_char = value[0]
        value = value[1:-1].strip()
        if quote_char == '"':
            value = value.replace('""', '"')
    return value


def parse_generated_tests(raw_csv, use_case, use_case_context, num_tests):
    cleaned_csv = clean_generated_csv_text(raw_csv)
    rows = [line.strip() for line in cleaned_csv.splitlines() if line.strip()]

    if not rows:
        raise ValueError("Generated CSV is empty.")

    header_id, header_input = split_generated_line(rows[0], "header")
    normalized_headers = [
        normalize_generated_header(header_id),
        normalize_generated_header(header_input),
    ]
    missing_columns = {"test_id", "input"} - set(normalized_headers)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Generated CSV is missing required column(s): {missing}")

    data_rows = rows[1:]

    if len(data_rows) != num_tests:
        raise ValueError(f"Generated CSV contained {len(data_rows)} prompt(s), expected {num_tests}.")

    prefix = slugify_use_case(use_case)
    tests = []
    for index, line in enumerate(data_rows, start=1):
        _, prompt = split_generated_line(line, f"row {index}")
        if not prompt:
            raise ValueError(f"Generated CSV row {index} has an empty input.")
        tests.append(
            {
                "test_id": f"{prefix}_{index:03d}",
                "use_case": use_case,
                "use_case_context": use_case_context or "",
                "input": prompt,
            }
        )

    return tests
