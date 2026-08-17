#!/usr/bin/env python3

import argparse
import json
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from explanation_validator import parse_and_validate
from render_explanation_prompt import iter_rendered_prompts


OLLAMA_URL = "http://localhost:11434/api/generate"
MAX_RAW_RESPONSE_CHARS = 12000


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def truncate_text(value: str, max_chars: int = MAX_RAW_RESPONSE_CHARS) -> str:
    if value is None:
        return ""
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "\n...[truncated]"


def should_retry(category: str, attempt: int, max_retries: int, retry_on: List[str]) -> bool:
    return attempt < max_retries and category in retry_on

# sends the prompt to the local Ollama server, model: gemma2:9b
# stream false meaning Ollama returns the complete answer at once
# returns raw_response(explanation produces by Gemma) & ollama_metadata(token counts and other ollama info)
def call_ollama(
    model: str, 
    prompt: str,
    generation_config: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": generation_config.get("temperature", 0.1),
            "top_p": generation_config.get("top_p", 0.9),
            "num_predict": generation_config.get("max_tokens", 700),
        },
    }

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    timeout = generation_config.get("timeout_seconds", 120)

    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_data = json.loads(response.read().decode("utf-8"))

    return response_data.get("response", ""), response_data


def build_retry_prompt(original_prompt: str, invalid_response: str, errors: List[str]) -> str:
    return """The previous response was invalid.

Validation errors:
{errors}

Previous response:
{invalid_response}

Return the corrected answer as valid JSON only.
Do not add markdown.
Do not add repair instructions.
Use the original input record below and preserve the required schema.

Original prompt:
{original_prompt}
""".format(
        errors="\n".join("- " + error for error in errors),
        invalid_response=invalid_response,
        original_prompt=original_prompt,
    )


def classify_validation_error(errors: List[str]) -> str:
    if errors and errors[0].startswith("invalid_json"):
        return "invalid_json"
    return "schema_validation_error"

# controls the entire process for one prompt
# calls ollama (call_ollama) and then receives the raw response
def explain_one(
    prompt: str,
    config: Dict[str, Any],
    schema_path: str
) -> Dict[str, Any]:
    model = config["models"]["primary"]
    generation_config = config.get("generation", {})
    retry_config = config.get("retry", {})
    max_retries = int(retry_config.get("max_retries", 0))
    retry_on = retry_config.get("retry_on", [])

    current_prompt = prompt
    raw_response = ""
    ollama_metadata: Dict[str, Any] = {}
    errors: List[str] = []
    final_category = None
    actual_attempts = 0

    start = time.time()

    for attempt in range(max_retries + 1):
        actual_attempts = attempt + 1
        try:
            raw_response, ollama_metadata = call_ollama(
                model=model,
                prompt=current_prompt,
                generation_config=generation_config,
            )

            valid, explanation, errors = parse_and_validate(raw_response, schema_path)

            if valid:
                return {
                    "status": "success",
                    "explanation_json": explanation,
                    "raw_response": truncate_text(raw_response),
                    "validation_errors": [],
                    "attempts": attempt + 1,
                    "latency_ms": int((time.time() - start) * 1000),
                    "tokens_input": ollama_metadata.get("prompt_eval_count"),
                    "tokens_output": ollama_metadata.get("eval_count"),
                    "error": None,
                }

            final_category = classify_validation_error(errors)

            if should_retry(final_category, attempt, max_retries, retry_on):
                current_prompt = build_retry_prompt(prompt, raw_response, errors)
                continue

            break

        except (TimeoutError, socket.timeout) as e:
            final_category = "timeout"
            ollama_metadata = {}
            errors = ["timeout: {}".format(e)]

            if should_retry(final_category, attempt, max_retries, retry_on):
                time.sleep(1)
                continue

            break

        except urllib.error.URLError as e:
            final_category = "model_unavailable"
            ollama_metadata = {}
            errors = ["model_unavailable: {}".format(e)]

            if should_retry(final_category, attempt, max_retries, retry_on):
                time.sleep(1)
                continue

            break

        except Exception as e:
            final_category = "runtime_error"
            ollama_metadata = {}
            errors = ["runtime_error: {}: {}".format(type(e).__name__, e)]
            break

    return {
        "status": "failed",
        "failure_category": final_category,
        "explanation_json": None,
        "raw_response": truncate_text(raw_response),
        "validation_errors": errors,
        "attempts": actual_attempts,
        "latency_ms": int((time.time() - start) * 1000),
        "tokens_input": ollama_metadata.get("prompt_eval_count"),
        "tokens_output": ollama_metadata.get("eval_count"),
        "error": "; ".join(errors),
    }


def build_input_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "notebook_execution_id": record.get("notebook_execution_id"),
        "error_type": record.get("error_type"),
        "error_message": record.get("error_message"),
        "failing_module": record.get("failing_module"),
        "original_subtype": record.get("original_subtype"),
        "refined_subtype": record.get("refined_subtype"),
        "scope_status": record.get("scope_status"),
        "exclusion_reason": record.get("exclusion_reason"),
        "split": record.get("split"),
        "classifier_confidence": record.get("confidence"),
        "root_cause_hint": record.get("root_cause_hint"),
        "context_status": record.get("context_status"),
        "error_cell_index": record.get("error_cell_index"),
    }


def build_llm_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "llm_model": config["models"]["primary"],
        "prompt_strategy": config["prompt"]["strategy"],
        "prompt_template": config["prompt"]["template"],
        "prompt_version": config["prompt"]["version"],
    }

# saves the explanation result
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM explanations over i2-classified dependency errors."
    )
    parser.add_argument("--config", default="config/llm_explainer.yaml")
    parser.add_argument("--input", default="data/context-classification/dependency_error_contexts.jsonl")
    parser.add_argument("--output")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    template_path = "prompts/{}.txt".format(config["prompt"]["template"])
    schema_path = config["output"]["schema"]
    output_path = Path(args.output or config["output"]["path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_id = "i3-{}".format(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))

    processed = 0

    if args.dry_run:
        for item in iter_rendered_prompts(args.input, template_path, limit=None):
            if item["index"] < args.start_index:
                continue
            if args.limit is not None and processed >= args.limit:
                break

            print("=" * 80)
            print("index:", item["index"])
            print("error:", item["error"])
            print(item["prompt"][:2000] if item["prompt"] else "")
            processed += 1
        return

    mode = "w" if args.overwrite else "a"

    with output_path.open(mode, encoding="utf-8") as out:
        for item in iter_rendered_prompts(args.input, template_path, limit=None):
            if item["index"] < args.start_index:
                continue
            if args.limit is not None and processed >= args.limit:
                break

            if item["error"]:
                result = {
                    "run_id": run_id,
                    "created_at": utc_now(),
                    "index": item["index"],
                    "input": build_input_metadata(item["record"]) if item.get("record") else None,
                    "llm": build_llm_metadata(config),
                    "explanation_result": {
                        "status": "render_failed",
                        "failure_category": "render_failed",
                        "error": item["error"],
                    },
                }
            else:
                record = item["record"]
                explanation_result = explain_one(item["prompt"], config, schema_path)

                result = {
                    "run_id": run_id,
                    "created_at": utc_now(),
                    "index": item["index"],
                    "input": build_input_metadata(record),
                    "llm": build_llm_metadata(config),
                    "explanation_result": explanation_result,
                }

            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()

            status = result["explanation_result"]["status"]
            print("[{}] index={} status={}".format(utc_now(), item["index"], status))

            processed += 1


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, json.JSONDecodeError, yaml.YAMLError) as e:
        print("ERROR: {}".format(e))
        raise SystemExit(1)
