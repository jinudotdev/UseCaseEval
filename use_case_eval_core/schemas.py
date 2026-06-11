OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

GENERATED_QUESTIONS_OUTPUT = "generated_questions.csv"
JUDGE_TESTS_OUTPUT = "generated_judge_tests.csv"
JUDGE_QUESTIONS_OUTPUT = "generated_judge_questions.csv"
JUDGE_1_RESULTS_OUTPUT = "judge_1_results.csv"
MODEL_RETURNS_OUTPUT = ".\\generated_model_returns.csv"
JUDGE_SCORES_OUTPUT = ".\\generated_judge_scores.csv"
FINAL_RESULTS_OUTPUT = ".\\final_results.csv"

GENERATED_QUESTIONS_COLUMNS = [
    "test_id",
    "input",
]

MODEL_RESULT_COLUMNS = [
    "test_id",
    "use_case",
    "input",
    "model_name",
    "model_response",
    "latency_ms",
    "tokens_per_second",
    "judge_1_model",
    "judge_1_score",
    "judge_1_reason",
    "judge_1_pass_fail",
    "judge_2_model",
    "judge_2_score",
    "judge_2_reason",
    "judge_2_pass_fail",
    "human_score",
    "human_notes",
]

FINAL_RESULTS_COLUMNS = MODEL_RESULT_COLUMNS

MODEL_RETURNS_COLUMNS = [
    "test_id",
    "use_case",
    "input",
    "model_name",
    "model_response",
    "latency_ms",
    "tokens_per_second",
]

JUDGE_SCORES_COLUMNS = [
    "test_id",
    "use_case",
    "input",
    "model_name",
    "judge_model",
    "judge_score",
    "judge_reason",
    "judge_pass_fail",
]

JUDGE_TEST_COLUMNS = [
    "judge_test_id",
    "source_test_id",
    "use_case",
    "user_input",
    "tested_model_name",
    "tested_model_response",
]

JUDGE_QUESTION_COLUMNS = [
    "test_id",
    "input",
    "expected_behavior",
    "judge_role",
    "judge_standard",
    "judge_rubric",
    "judge_output_format",
]

JUDGE_1_RESULT_COLUMNS = [
    "judge_test_id",
    "source_test_id",
    "tested_model_name",
    "judge_1_model",
    "judge_1_score",
    "judge_1_reason",
    "judge_1_pass_fail",
]

JUDGE_2_RESULT_COLUMNS = [
    "judge_test_id",
    "source_test_id",
    "tested_model_name",
    "judge_2_model",
    "judge_2_score",
    "judge_2_reason",
    "judge_2_pass_fail",
]
