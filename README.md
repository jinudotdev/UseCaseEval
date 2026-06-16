# UseCaseEval
 
<img width="722" height="752" alt="use_case_eval" src="https://github.com/user-attachments/assets/53434c43-075c-410f-99f0-b59e0dccab03" />

## TL;DR

UseCaseEval answers the question: **“Which local model will work best for this job?”**

When hardware is limited, especially on mobile devices, we need to choose a model that fits the available resources while still performing the task reliably. The goal is to find the smallest model that is good enough for the job.

### Quick start

You need:

* Python 3.11 or newer
* Ollama installed and running
* At least one Ollama model downloaded

Clone this repository, open PowerShell in the project folder, install the requirements, and launch the GUI:

```powershell
pip install -r requirements.txt
python -m use_case_eval_gui.app
```


---

## What is UseCaseEval?

UseCaseEval is a local LLM evaluation pipeline for testing small language models against specific use cases.

It is designed for local-first AI experimentation with:

* Python
* Ollama
* GGUF/local models
* CSV-based evaluation
* generated test questions
* generated judge rubrics
* human-reviewable outputs

The project is meant to help compare small models for practical app scenarios, especially where local inference matters.

Example use cases:

* Senior voice assistant
* Pet health assistant
* Local mobile assistant
* Reminder assistant
* Offline companion app
* Financial news summarizer
* Cooking assistant
* Travel assistant

---

## Why this matters

Phones and consumer devices are becoming increasingly capable of running small local language models. That makes local-first AI apps more realistic, especially for use cases where privacy, offline access, low latency, or avoiding API costs are important.

But running a model locally is only part of the problem.

A model also needs to be good enough for the specific job it is being used for.

A small model may be useful for one task and unsafe or unreliable for another. For example, a model that works well for casual chat may fail when asked to handle reminders, health-related questions, voice-assistant responses, or structured JSON output.

UseCaseEval is meant to help test that practical fit.

Instead of asking only:

> Which model is best?

this project asks:

> Which model is good enough for this specific use case?

It helps compare:

* answer quality
* safety
* hallucination risk
* response length
* voice-readability
* speed
* local model suitability
* tool-boundary awareness

This is especially relevant for mobile and edge AI, where developers often need to choose smaller models that balance quality, speed, memory use, and safety.

---

## Current workflow

UseCaseEval currently creates five main generated CSV files:

```text
generated_questions.csv
generated_judge_questions.csv
generated_model_returns.csv
generated_judge_scores.csv
final_results.csv
```

Each file represents one clean step in the evaluation pipeline.

---

## Default usage

Run:

```powershell
python .\use_case_eval.py
```

The app asks:

```text
What use case would you like to evaluate?
Briefly describe the intended assistant in 1-4 sentences.
Include its target users, expected tasks, interface, and important limitations.
Press Enter to continue without additional context:
```

Example answer:

```text
child toy companion
```

Then it automatically runs the full pipeline and writes:

```text
generated_questions.csv
generated_judge_questions.csv
generated_model_returns.csv
generated_judge_scores.csv
final_results.csv
```

Default models and settings:

```text
num_tests: 10
generator_model: qwen35-9b
judge_question_generator_model: qwen35-9b
judge_model: qwen35-9b
tested models: qwen25-15b-q4,llama32-1b-q4,tinyllama-11b-q4,smollm2-17b-q4,qwen25-05b-q8
max_tokens: 220
```

To skip the interactive prompt:

```powershell
python .\use_case_eval.py --use-case "child toy companion"
```

Add `--use-case-context` to make generated tests match the intended assistant more closely.

You can also choose a config file explicitly:

```powershell
python .\use_case_eval.py --config .\use_case_eval_config.toml
```

---

## Configuration

UseCaseEval reads defaults from `use_case_eval_config.toml`. If the file is missing, UseCaseEval prints a message and uses the same built-in defaults.

```toml
[models]
frontier = "qwen35-9b"
judge_1 = "qwen35-9b"
judge_2 = ""

evaluated = [
  "qwen25-15b-q4",
  "llama32-1b-q4",
  "tinyllama-11b-q4",
  "smollm2-17b-q4",
  "qwen25-05b-q8"
]

[run]
num_tests = 10
max_tokens = 220
judge_pass_threshold = 4

[paths]
generated_questions = "generated_questions.csv"
generated_judge_questions = "generated_judge_questions.csv"
generated_model_returns = "generated_model_returns.csv"
generated_judge_scores = "generated_judge_scores.csv"
final_results = "final_results.csv"
```

Model roles:

* `frontier` generates test questions and judge-question rubrics.
* `judge_1` scores model returns into `generated_judge_scores.csv`.
* `judge_2` is optional and may be blank for now.
* `evaluated` is the list of smaller models being tested.

To change evaluated models, edit the `evaluated` list in `[models]`.

When Ollama is running, UseCaseEval checks configured model names against local `/api/tags` before model-heavy workflows run. Missing evaluated models are skipped with a warning. Missing required models, such as `frontier` or `judge_1`, stop the run with a clear error. If Ollama is not running, UseCaseEval prints a warning and continues without model existence validation.

CLI flags override config values. For example, `--models`, `--generator-model`, `--judge-model`, `--num-tests`, `--max-tokens`, and path flags all take precedence over `use_case_eval_config.toml`.

---

## Describe the model you want to evaluate

A use-case name alone may be ambiguous. For example, "senior medical assistant" could mean an experienced clinical assistant or a non-clinical assistant for older adults.

Use `--use-case-context` to describe:

* who the model serves
* whether it is a voice, chat, mobile, or desktop assistant
* what tasks it should perform
* what tools or data it can access
* what it must not do

Use case:

```text
beginner computer helper
```

Context:

```text
A small offline voice assistant for people who are new to Windows. It explains basic computer concepts, gives short spoken instructions, and helps interpret simple error messages. It cannot access the internet or control the computer.
```

Example:

```powershell
python .\use_case_eval.py `
  --generate-tests `
  --use-case "beginner computer helper" `
  --use-case-context "A small offline voice assistant for people who are new to Windows. It explains basic computer concepts, gives short spoken instructions, and helps interpret simple error messages. It cannot access the internet or control the computer." `
  --num-tests 3
```

If no context is supplied, UseCaseEval still runs, but generated tests may interpret the use case differently than intended.

UseCaseEval stores `use_case_context` in generated CSV artifacts for recordkeeping. The context defines the intended users, assistant interface, expected tasks, available tools, and important limitations. Older records with blank context may be less precise or reproducible because the use-case name alone may not fully describe what was evaluated.

The context can also describe assumed host-application capabilities. For example, it may say that the assistant can remember approved preferences or create recurring reminders through the host application. In that case, evaluated models are expected to respond as though the host application provides those capabilities:

```text
User: Remember that I prefer short answers.
Assistant: Got it. I'll remember that you prefer short answers.
```

UseCaseEval itself remains stateless. It does not store memories, schedule reminders, call tools, or verify that actions occurred. Every generated test remains independent. Capabilities not listed in the context must not be claimed:

```text
User: Can you call my daughter?
Assistant: I'm sorry, but I can't access your contacts or place calls.
```

---

## 1. Generate test questions

`generated_questions.csv` contains the questions that each tested model will answer.

Schema:

```csv
test_id,use_case,use_case_context,input
```

Example:

```csv
pet_health_assistant_001,My cat vomited twice today; is this normal?
```

Command:

```powershell
python .\use_case_eval.py `
  --generate-tests `
  --use-case "pet health assistant" `
  --use-case-context "A small offline voice assistant that gives concise general pet-care information and encourages veterinary help when appropriate. It cannot diagnose pets, access live records, or contact a veterinarian." `
  --num-tests 3 `
  --generator-model qwen35-9b `
  --generated-questions .\generated_questions.csv
```

If `--use-case` is omitted, the app prompts for it.

---

## 2. Generate judge questions

`generated_judge_questions.csv` contains the expected behavior and scoring rubric for each test question.

Schema:

```csv
test_id,use_case,use_case_context,input,expected_behavior,judge_role,judge_standard,judge_rubric,judge_output_format
```

The judge rubric is generated dynamically from the use case and question.

Each rubric uses a consistent 1-to-5 scoring structure:

```text
Score 1 [Critical Failure]: ...
Score 2 [Poor / Incomplete]: ...
Score 3 [Acceptable / Basic]: ...
Score 4 [Good / Professional]: ...
Score 5 [Excellent]: ...
```

Command:

```powershell
python .\use_case_eval.py `
  --use-case "pet health assistant" `
  --export-judge-questions `
  --judge-question-generator-model qwen35-9b `
  --debug-judge-question-generator
```

This creates:

```text
generated_judge_questions.csv
```

The debug flag is useful during development because it shows the raw local model response and parsing behavior.

---

## 3. Generate model returns

`generated_model_returns.csv` contains the raw answers from the smaller tested models.

Schema:

```csv
test_id,use_case,use_case_context,input,model_name,model_response,tokens_per_second
```

This file intentionally does not include judge scores yet. It is only the model-answer layer.

Command:

```powershell
python .\use_case_eval.py `
  --input .\generated_questions.csv `
  --use-case "pet health assistant" `
  --models "qwen25-15b-q4,llama32-1b-q4,tinyllama-11b-q4,smollm2-17b-q4,qwen25-05b-q8" `
  --generate-model-returns `
  --model-returns-output .\generated_model_returns.csv `
  --max-tokens 220
```

This creates:

```text
generated_model_returns.csv
```

---

## 4. Generate judge scores

`generated_judge_scores.csv` contains one score row for each model answer.

It combines:

* the judge instructions from `generated_judge_questions.csv`
* the model answers from `generated_model_returns.csv`
* a local judge model, such as `qwen35-9b`

Schema:

```csv
test_id,use_case,use_case_context,input,model_name,judge_slot,judge_model,judge_score,judge_reason,judge_pass_fail
```

Command:

```powershell
python .\use_case_eval.py `
  --generate-judge-scores `
  --judge-model qwen35-9b `
  --judge-questions-input .\generated_judge_questions.csv `
  --model-returns-input .\generated_model_returns.csv `
  --judge-scores-output .\generated_judge_scores.csv `
  --judge-pass-threshold 4 `
  --debug-judge-scores
```

This creates:

```text
generated_judge_scores.csv
```

The debug flag shows the judge request, raw Ollama response, extracted JSON text, and parse errors.

---

## 5. Generate final results

`final_results.csv` combines model answers and generated judge scores into the final review table.

It keeps Judge 2 and human review fields blank for now.

Schema:

```csv
test_id,use_case,use_case_context,input,model_name,model_response,tokens_per_second,judge_1_model,judge_1_score,judge_1_reason,judge_1_pass_fail,judge_2_model,judge_2_score,judge_2_reason,judge_2_pass_fail,combined_judge_result,human_score,human_notes
```

Command:

```powershell
python .\use_case_eval.py `
  --generate-final-results `
  --model-returns-input .\generated_model_returns.csv `
  --judge-scores-input .\generated_judge_scores.csv `
  --final-results-output .\final_results.csv
```

This creates:

```text
final_results.csv
```

---

## Full workflow

The full pipeline is:

```text
generated_questions.csv
        ↓
generated_judge_questions.csv
        ↓
generated_model_returns.csv
        ↓
generated_judge_scores.csv
        ↓
final_results.csv
```

The final result file will merge model answers, judge scores, and blank human review columns.

Final schema:

```csv
test_id,use_case,use_case_context,input,model_name,model_response,tokens_per_second,judge_1_model,judge_1_score,judge_1_reason,judge_1_pass_fail,judge_2_model,judge_2_score,judge_2_reason,judge_2_pass_fail,combined_judge_result,human_score,human_notes
```

---

## Why use a stronger local model as the generator/judge helper?

The project assumes you may have one stronger local model that runs well on your main machine, and several smaller models that you want to test for practical deployment.

For example:

* A stronger model on a desktop or laptop can generate questions and rubrics.
* Smaller models can be tested as candidates for phones, mobile apps, or edge devices.

This lets the stronger model act as an evaluation assistant while the smaller models are evaluated for real use.

The goal is not to prove that the strongest model is best. The goal is to find the smallest local model that is good enough for a specific job.

---

## Requirements

* Python 3.11 or newer
* Ollama running locally
* Local Ollama models already installed

Example models used during development:

```text
qwen35-9b
qwen25-15b-q4
llama32-1b-q4
tinyllama-11b-q4
smollm2-17b-q4
qwen25-05b-q8
```

---

## Setup

Clone the repo:

```powershell
git clone https://github.com/jinudotdev/UseCaseEval.git
cd UseCaseEval
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Make sure Ollama is running:

```powershell
ollama list
```

---

## Project structure

```text
UseCaseEval/
├─ use_case_eval.py
├─ requirements.txt
├─ use_case_eval_config.toml
├─ README.md
├─ use_case_eval_core/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ csv_utils.py
│  ├─ judge_question_generation.py
│  ├─ judge_scores_generation.py
│  ├─ final_results_generation.py
│  ├─ model_returns_generation.py
│  ├─ ollama_client.py
│  ├─ question_generation.py
│  └─ schemas.py
```

---

## Generated files

Generated CSV files are ignored by Git because they are local experiment outputs.

Ignored generated files include:

```text
generated_questions.csv
generated_judge_questions.csv
generated_model_returns.csv
generated_judge_scores.csv
results_current.csv
final_results.csv
judge_1_results.csv
judge_2_results.csv
```

---

## What this project demonstrates

This project demonstrates:

* Running local Ollama models from Python
* Building a CLI-based evaluation workflow
* Generating use-case-specific test questions
* Generating dynamic judge rubrics
* Comparing multiple small local models on the same questions
* Exporting model responses to CSV
* Debugging structured JSON output from local reasoning models
* Refactoring a single script into a modular Python package
* Using Codex-assisted development while reviewing and testing each change manually

---

## Status

This project is under active development.

Currently implemented:

* Question generation
* Dynamic judge-question generation
* Ordered 1-to-5 generated judge rubrics
* Standalone model-return generation
* Standalone judge-score generation
* Standalone final-results generation
* Modular Python package structure

Planned next steps:

* Judge 2 scoring workflow
* Human review columns
* Cleaner subcommands for each workflow step

---

## Notes

UseCaseEval is an experimental learning project. It is not intended to provide medical, legal, financial, or professional advice.

Any domain-specific examples are used only to test model behavior and evaluation workflows.
