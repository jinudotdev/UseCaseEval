import copy
import os
import sys
import tomllib

from .ollama_client import requests
from .schemas import OLLAMA_TAGS_URL


DEFAULT_CONFIG_PATH = "use_case_eval_config.toml"

INTERNAL_DEFAULT_CONFIG = {
    "models": {
        "frontier": "qwen35-9b",
        "judge_1": "qwen35-9b",
        "judge_2": "",
        "evaluated": [
            "qwen25-15b-q4",
            "llama32-1b-q4",
            "tinyllama-11b-q4",
            "smollm2-17b-q4",
            "qwen25-05b-q8",
        ],
    },
    "run": {
        "num_tests": 10,
        "max_tokens": 220,
        "judge_pass_threshold": 4,
    },
    "paths": {
        "generated_questions": "generated_questions.csv",
        "generated_judge_questions": "generated_judge_questions.csv",
        "generated_model_returns": "generated_model_returns.csv",
        "generated_judge_scores": "generated_judge_scores.csv",
        "final_results": "final_results.csv",
    },
}


def load_config(config_path=None):
    config_path = config_path or DEFAULT_CONFIG_PATH
    config = copy.deepcopy(INTERNAL_DEFAULT_CONFIG)

    if not os.path.exists(config_path):
        print(
            f"Config file not found: {config_path}. Using built-in defaults.",
            file=sys.stderr,
        )
        return config

    with open(config_path, "rb") as config_file:
        loaded_config = tomllib.load(config_file)

    merge_config(config, loaded_config)
    normalize_config(config)
    return config


def merge_config(base_config, override_config):
    if not isinstance(override_config, dict):
        raise ValueError("Config file must contain TOML tables.")

    for section_name in ("models", "run", "paths"):
        section = override_config.get(section_name)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise ValueError(f"Config section [{section_name}] must be a TOML table.")
        base_config[section_name].update(section)


def normalize_config(config):
    models = config["models"]
    run = config["run"]
    paths = config["paths"]

    for key in ("frontier", "judge_1", "judge_2"):
        value = models.get(key, "")
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise ValueError(f"Config models.{key} must be a string.")
        models[key] = value.strip()

    evaluated = models.get("evaluated", [])
    if not isinstance(evaluated, list):
        raise ValueError("Config models.evaluated must be a list of strings.")
    normalized_evaluated = []
    for model_name in evaluated:
        if not isinstance(model_name, str):
            raise ValueError("Config models.evaluated must contain only strings.")
        model_name = model_name.strip()
        if model_name:
            normalized_evaluated.append(model_name)
    models["evaluated"] = normalized_evaluated

    for key in ("num_tests", "max_tokens", "judge_pass_threshold"):
        value = run.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Config run.{key} must be an integer.")
        run[key] = value

    for key in (
        "generated_questions",
        "generated_judge_questions",
        "generated_model_returns",
        "generated_judge_scores",
        "final_results",
    ):
        value = paths.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Config paths.{key} must be a non-empty string.")
        paths[key] = value.strip()


def apply_config_defaults(args, config):
    models = config["models"]
    run = config["run"]
    paths = config["paths"]

    set_default(args, "generator_model", models["frontier"])
    set_default(args, "judge_question_generator_model", models["frontier"])
    set_default(args, "judge_model", models["judge_1"])
    set_default(args, "judge_1_model", models["judge_1"])
    set_default(args, "judge_2_model", models["judge_2"])
    set_default(args, "models", ",".join(models["evaluated"]))

    set_default(args, "num_tests", run["num_tests"])
    set_default(args, "max_tokens", run["max_tokens"])
    set_default(args, "judge_pass_threshold", run["judge_pass_threshold"])
    set_default(args, "judge_1_threshold", run["judge_pass_threshold"])

    set_default(args, "generated_questions", paths["generated_questions"])
    set_default(args, "judge_questions_input", paths["generated_judge_questions"])
    set_default(args, "judge_questions_output", paths["generated_judge_questions"])
    set_default(args, "model_returns_input", paths["generated_model_returns"])
    set_default(args, "model_returns_output", paths["generated_model_returns"])
    set_default(args, "judge_scores_input", paths["generated_judge_scores"])
    set_default(args, "judge_scores_output", paths["generated_judge_scores"])
    set_default(args, "final_results_output", paths["final_results"])


def set_default(args, attribute_name, value):
    if getattr(args, attribute_name, None) is None:
        setattr(args, attribute_name, value)


def fetch_installed_ollama_models():
    if requests is None:
        print(
            "Warning: requests is not installed; skipping Ollama model validation.",
            file=sys.stderr,
        )
        return None

    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.ConnectionError:
        print(
            "Warning: Could not connect to Ollama for model validation; continuing without validation.",
            file=sys.stderr,
        )
        return None
    except requests.exceptions.RequestException as error:
        print(
            f"Warning: Could not validate Ollama models: {error}",
            file=sys.stderr,
        )
        return None
    except ValueError as error:
        print(
            f"Warning: Ollama /api/tags returned invalid JSON: {error}",
            file=sys.stderr,
        )
        return None

    installed_models = set()
    for model in data.get("models", []):
        if not isinstance(model, dict):
            continue
        for key in ("name", "model"):
            model_name = model.get(key)
            if isinstance(model_name, str) and model_name.strip():
                installed_models.add(model_name.strip())

    return installed_models


def model_is_installed(model_name, installed_models):
    if not model_name:
        return False
    if model_name in installed_models:
        return True
    if f"{model_name}:latest" in installed_models:
        return True
    return any(installed_model.split(":", 1)[0] == model_name for installed_model in installed_models)


def validate_model_roles(
    required_models=None,
    optional_models=None,
    evaluated_models=None,
    require_evaluated=False,
):
    required_models = required_models or []
    optional_models = optional_models or []
    evaluated_models = evaluated_models or []

    errors = []
    for role_name, model_name in required_models:
        if not model_name:
            errors.append(f"{role_name} model is required but is blank.")

    if require_evaluated and not evaluated_models:
        errors.append("At least one evaluated model is required.")

    for role_name, model_name in optional_models:
        if not model_name:
            print(f"Warning: optional {role_name} model is blank; skipping {role_name}.", file=sys.stderr)

    if errors:
        raise ValueError(" ".join(errors))

    installed_models = fetch_installed_ollama_models()
    if installed_models is None:
        return {
            "evaluated": list(evaluated_models),
            "optional": {role_name: model_name for role_name, model_name in optional_models},
        }

    for role_name, model_name in required_models:
        if not model_is_installed(model_name, installed_models):
            errors.append(
                f"Required {role_name} model '{model_name}' is not installed in Ollama."
            )

    optional = {}
    for role_name, model_name in optional_models:
        if not model_name:
            optional[role_name] = ""
            continue
        if model_is_installed(model_name, installed_models):
            optional[role_name] = model_name
        else:
            print(
                f"Warning: optional {role_name} model '{model_name}' is not installed; skipping {role_name}.",
                file=sys.stderr,
            )
            optional[role_name] = ""

    filtered_evaluated = []
    for model_name in evaluated_models:
        if model_is_installed(model_name, installed_models):
            filtered_evaluated.append(model_name)
        else:
            print(
                f"Warning: evaluated model '{model_name}' is not installed; skipping it.",
                file=sys.stderr,
            )

    if require_evaluated and not filtered_evaluated:
        errors.append("No evaluated models are available after Ollama model validation.")

    if errors:
        raise ValueError(" ".join(errors))

    return {
        "evaluated": filtered_evaluated,
        "optional": optional,
    }
