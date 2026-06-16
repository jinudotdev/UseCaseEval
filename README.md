# UseCaseEval

<img width="722" height="752" alt="UseCaseEval GUI" src="https://github.com/user-attachments/assets/53434c43-075c-410f-99f0-b59e0dccab03" />

## TL;DR

UseCaseEval answers the question: **“Which local model will work best for this job?”**

When hardware is limited, especially on mobile devices, the goal is to find the smallest local model that can still perform a specific task reliably.

## Quick start

You need:

* Python 3.11 or newer
* Ollama installed and running
* At least one Ollama model downloaded

Clone the repository:

```powershell
git clone https://github.com/jinudotdev/UseCaseEval.git
cd UseCaseEval
```

Install the requirements:

```powershell
pip install -r requirements.txt
```

Launch the GUI:

```powershell
python -m use_case_eval_gui.app
```

## How it works

UseCaseEval uses stronger local models to:

1. Generate realistic test questions for a use case.
2. Generate scoring rubrics for those questions.
3. Run selected candidate models against the questions.
4. Score each response with one or two local judge models.
5. Export the results to CSV for review.

Everything runs locally through Ollama. No paid API is required.

## GUI workflow

The GUI automatically scans Ollama for installed models.

It lets you choose:

* The use case to evaluate
* Optional context describing the intended assistant
* A Frontier model for generating questions and rubrics
* One or two Judge models
* One or more models to evaluate
* Number of sample questions
* Maximum response tokens
* Judge pass threshold

The largest installed models are suggested for the Frontier and Judge roles. These suggestions can be changed before running the evaluation.

## Use-case context

The optional use-case context describes:

* Intended users
* Expected tasks
* Interface type
* Available capabilities
* Important limitations

Example:

```text
A small offline voice assistant for seniors. It answers general questions,
provides short computer-help instructions, and can create reminders through
its host application. It cannot browse the internet, place calls, or access
private accounts.
```

The context is authoritative during question generation and judging.

UseCaseEval itself remains stateless. It does not create reminders, store memories, call tools, or verify that actions occurred. It only evaluates whether model responses are appropriate for the capabilities described in the context.

## Output files

A full evaluation creates:

```text
generated_questions.csv
generated_judge_questions.csv
generated_model_returns.csv
generated_judge_scores.csv
final_results.csv
```

The main output is:

```text
final_results.csv
```

It contains:

* Test question
* Evaluated model
* Model response
* Tokens per second
* Judge scores
* Judge explanations
* Pass/fail results
* Empty human-review fields

## Example use cases

* Senior voice assistant
* Local mobile assistant
* Offline companion
* Reminder assistant
* Beginner computer helper
* Cooking assistant
* Travel assistant
* Structured-output model
* Pet-care information assistant

## Model roles

### Frontier model

Generates evaluation questions and judge rubrics.

A capable model is recommended because the quality of the evaluation depends heavily on the quality of the generated tests.

### Judge models

One or two local models score each candidate response.

Using two judges can help expose disagreements or inconsistent scoring.

### Evaluated models

These are the smaller candidate models being compared for the intended use case.

## Scoring

Responses are graded from 1 to 5:

```text
1 — Critical failure
2 — Poor or incomplete
3 — Acceptable
4 — Good
5 — Excellent
```

The pass threshold is configurable. The GUI defaults to:

```text
3
```

Judge priorities include:

1. Factual correctness
2. Completion of the user’s request
3. Compliance with the supplied use-case context
4. Safety
5. Clarity

Concise correct answers may receive a score of 5. Models are not required to add greetings, unnecessary explanations, offers of additional help, or capability disclaimers unless the situation requires them.

## CLI usage

The original command-line interface remains available:

```powershell
python .\use_case_eval.py
```

You can also supply options directly:

```powershell
python .\use_case_eval.py `
  --use-case "beginner computer helper" `
  --use-case-context "A small offline assistant that gives short Windows instructions and cannot control the computer." `
  --num-tests 3
```

View all CLI options:

```powershell
python .\use_case_eval.py --help
```

## Configuration

Default CLI settings can be stored in:

```text
use_case_eval_config.toml
```

GUI selections are passed into the same shared evaluation pipeline without changing the CLI workflow.

## Project structure

```text
UseCaseEval/
├─ use_case_eval.py
├─ use_case_eval_config.toml
├─ requirements.txt
├─ README.md
├─ tests/
├─ use_case_eval_core/
│  ├─ config.py
│  ├─ csv_utils.py
│  ├─ generate_question.py
│  ├─ generate_judge_question.py
│  ├─ generate_model_returns.py
│  ├─ generate_judge_scores.py
│  ├─ generate_final_results.py
│  ├─ ollama_client.py
│  └─ schemas.py
└─ use_case_eval_gui/
   ├─ __init__.py
   └─ app.py
```

## Requirements

* Windows, Linux, or another platform supported by Python and Ollama
* Python 3.11 or newer
* Ollama running locally
* One or more installed Ollama models

Check installed models with:

```powershell
ollama list
```

## Limitations

* Model-based judging is not perfectly objective.
* Results depend on the quality of the selected Frontier and Judge models.
* Tokens per second is a rough comparison metric.
* Actual performance on phones or edge devices should be measured directly on the target hardware.
* Generated evaluations should still be reviewed by a human when safety or reliability matters.

## Status

Implemented:

* Tkinter GUI
* Automatic Ollama model discovery
* Size-based Frontier and Judge suggestions
* Use-case and context-aware question generation
* Dynamic judge rubrics
* One- or two-judge scoring
* Multi-model evaluation
* CSV exports
* Human-review columns
* CLI workflow
* Automated tests

## License

This project is available under the MIT License. See `LICENSE` for details.

## Disclaimer

UseCaseEval is an experimental evaluation tool. It is not intended to provide medical, legal, financial, or other professional advice.

Domain-specific examples are used only to test model behavior.
