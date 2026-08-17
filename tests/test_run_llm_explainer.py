import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_llm_explainer


SCHEMA = str(ROOT / "schemas" / "explanation.schema.json")


def config():
    return {
        "models": {"primary": "gemma2:9b", "fallback": None},
        "generation": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 700,
            "timeout_seconds": 120
        },
        "retry": {
            "max_retries": 1,
            "retry_on": [
                "invalid_json",
                "schema_validation_error",
                "timeout",
                "model_unavailable"
            ]
        },
        "prompt": {
            "strategy": "few_shot",
            "template": "dependency_explanation_v1",
            "version": "i3_prompt_v1"
        }
    }


def valid_response():
    return {
        "summary": "The notebook failed because sklearn is missing.",
        "root_cause": "Python could not import sklearn in the execution environment.",
        "evidence": ["The error type is ModuleNotFoundError."],
        "failing_module": "sklearn",
        "explanation_confidence": "high",
        "limitations": "Only metadata is available."
    }


def test_explain_one_success_with_mocked_ollama(monkeypatch):
    def fake_call_ollama(model, prompt, generation_config):
        return json.dumps(valid_response()), {
            "prompt_eval_count": 10,
            "eval_count": 20
        }

    monkeypatch.setattr(run_llm_explainer, "call_ollama", fake_call_ollama)

    result = run_llm_explainer.explain_one("prompt", config(), SCHEMA)

    assert result["status"] == "success"
    assert result["attempts"] == 1
    assert result["tokens_input"] == 10
    assert result["tokens_output"] == 20
    assert result["explanation_json"]["failing_module"] == "sklearn"


def test_explain_one_retries_after_invalid_json(monkeypatch):
    calls = {"count": 0}

    def fake_call_ollama(model, prompt, generation_config):
        calls["count"] += 1
        if calls["count"] == 1:
            return "not json", {"prompt_eval_count": 5, "eval_count": 5}
        return json.dumps(valid_response()), {"prompt_eval_count": 10, "eval_count": 20}

    monkeypatch.setattr(run_llm_explainer, "call_ollama", fake_call_ollama)

    result = run_llm_explainer.explain_one("prompt", config(), SCHEMA)

    assert result["status"] == "success"
    assert result["attempts"] == 2
    assert calls["count"] == 2


def test_explain_one_runtime_error_attempt_count_is_one(monkeypatch):
    def fake_call_ollama(model, prompt, generation_config):
        raise RuntimeError("boom")

    monkeypatch.setattr(run_llm_explainer, "call_ollama", fake_call_ollama)

    result = run_llm_explainer.explain_one("prompt", config(), SCHEMA)

    assert result["status"] == "failed"
    assert result["failure_category"] == "runtime_error"
    assert result["attempts"] == 1

import urllib.error


def test_explain_one_retries_after_timeout(monkeypatch):
    calls = {"count": 0}

    def fake_call_ollama(model, prompt, generation_config):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimeoutError("slow model")
        return json.dumps(valid_response()), {"prompt_eval_count": 10, "eval_count": 20}

    monkeypatch.setattr(run_llm_explainer, "call_ollama", fake_call_ollama)
    monkeypatch.setattr(run_llm_explainer.time, "sleep", lambda _: None)

    result = run_llm_explainer.explain_one("prompt", config(), SCHEMA)

    assert result["status"] == "success"
    assert result["attempts"] == 2
    assert calls["count"] == 2


def test_explain_one_retries_after_model_unavailable(monkeypatch):
    calls = {"count": 0}

    def fake_call_ollama(model, prompt, generation_config):
        calls["count"] += 1
        if calls["count"] == 1:
            raise urllib.error.URLError("ollama unavailable")
        return json.dumps(valid_response()), {"prompt_eval_count": 10, "eval_count": 20}

    monkeypatch.setattr(run_llm_explainer, "call_ollama", fake_call_ollama)
    monkeypatch.setattr(run_llm_explainer.time, "sleep", lambda _: None)

    result = run_llm_explainer.explain_one("prompt", config(), SCHEMA)

    assert result["status"] == "success"
    assert result["attempts"] == 2
    assert calls["count"] == 2


def test_explain_one_fails_after_retry_exhaustion(monkeypatch):
    calls = {"count": 0}

    def fake_call_ollama(model, prompt, generation_config):
        calls["count"] += 1
        return "not json", {"prompt_eval_count": 5, "eval_count": 5}

    monkeypatch.setattr(run_llm_explainer, "call_ollama", fake_call_ollama)

    result = run_llm_explainer.explain_one("prompt", config(), SCHEMA)

    assert result["status"] == "failed"
    assert result["failure_category"] == "invalid_json"
    assert result["attempts"] == 2
    assert calls["count"] == 2


def test_retry_on_config_gates_retries(monkeypatch):
    cfg = config()
    cfg["retry"]["retry_on"] = []

    calls = {"count": 0}

    def fake_call_ollama(model, prompt, generation_config):
        calls["count"] += 1
        return "not json", {"prompt_eval_count": 5, "eval_count": 5}

    monkeypatch.setattr(run_llm_explainer, "call_ollama", fake_call_ollama)

    result = run_llm_explainer.explain_one("prompt", cfg, SCHEMA)

    assert result["status"] == "failed"
    assert result["attempts"] == 1
    assert calls["count"] == 1


def test_build_input_metadata_preserves_scope_fields():
    record = {
        "notebook_execution_id": 15,
        "error_type": "ImportError",
        "error_message": "libxcb.so.1: cannot open shared object file: No such file or directory",
        "failing_module": "libxcb.so.1",
        "original_subtype": "system_library",
        "refined_subtype": "system_library",
        "scope_status": "excluded",
        "exclusion_reason": "requires system library, outside pip-only scope",
        "split": "excluded",
        "confidence": "high",
        "root_cause_hint": "system_level_dependency",
        "context_status": "metadata_only",
        "error_cell_index": "8",
    }

    metadata = run_llm_explainer.build_input_metadata(record)

    assert metadata["scope_status"] == "excluded"
    assert metadata["exclusion_reason"] == "requires system library, outside pip-only scope"
    assert metadata["split"] == "excluded"


def test_build_input_metadata_preserves_scope_fields_for_usable_row():
    record = {
        "notebook_execution_id": 8,
        "error_type": "ModuleNotFoundError",
        "error_message": "No module named 'sklearn'",
        "failing_module": "sklearn",
        "original_subtype": "missing_package",
        "refined_subtype": "missing_package",
        "scope_status": "usable",
        "exclusion_reason": "",
        "split": "dev",
        "confidence": "high",
        "root_cause_hint": "import_distribution_name_mismatch",
        "context_status": "metadata_only",
        "error_cell_index": "5",
    }

    metadata = run_llm_explainer.build_input_metadata(record)

    assert metadata["scope_status"] == "usable"
    assert metadata["split"] == "dev"


def test_main_writes_nested_output_with_mocked_ollama(monkeypatch, tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    rows = [
        {
            "notebook_execution_id": 1,
            "error_type": "ModuleNotFoundError",
            "error_message": "No module named 'a'",
            "original_subtype": "missing_package",
            "refined_subtype": "missing_package",
            "confidence": "high",
            "root_cause_hint": "direct_missing_package",
            "failing_module": "a",
            "context_status": "metadata_only",
            "error_cell_index": "1",
            "prompt_context": {},
            "legacy_traceback_hint": None
        },
        {
            "notebook_execution_id": 2,
            "error_type": "ModuleNotFoundError",
            "error_message": "No module named 'sklearn'",
            "original_subtype": "missing_package",
            "refined_subtype": "missing_package",
            "confidence": "high",
            "root_cause_hint": "import_distribution_name_mismatch",
            "failing_module": "sklearn",
            "context_status": "metadata_only",
            "error_cell_index": "5",
            "prompt_context": {},
            "legacy_traceback_hint": None
        }
    ]

    input_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def fake_call_ollama(model, prompt, generation_config):
        return json.dumps(valid_response()), {"prompt_eval_count": 10, "eval_count": 20}

    monkeypatch.setattr(run_llm_explainer, "call_ollama", fake_call_ollama)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_llm_explainer.py",
            "--input", str(input_path),
            "--output", str(output_path),
            "--start-index", "1",
            "--limit", "1",
            "--overwrite"
        ]
    )

    run_llm_explainer.main()

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    row = json.loads(lines[0])
    assert row["index"] == 1
    assert row["input"]["notebook_execution_id"] == 2
    assert row["llm"]["llm_model"] == "gemma2:9b"
    assert row["explanation_result"]["status"] == "success"
    assert row["explanation_result"]["explanation_json"]["failing_module"] == "sklearn"


def test_main_does_not_filter_excluded_subtypes_from_explanation(monkeypatch, tmp_path):
    """The runner must explain every DEPENDENCY_ERROR row it is given,
    regardless of scope_status - repair eligibility is a downstream (i4)
    concern, not an explanation-time filter. See docs/architecture-note.md
    §7.1 and docs/prompts.md "i4 scope clarification"."""
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    rows = [
        {
            "notebook_execution_id": 8,
            "error_type": "ModuleNotFoundError",
            "error_message": "No module named 'sklearn'",
            "original_subtype": "missing_package",
            "refined_subtype": "missing_package",
            "scope_status": "usable",
            "exclusion_reason": "",
            "split": "dev",
            "confidence": "high",
            "root_cause_hint": "import_distribution_name_mismatch",
            "failing_module": "sklearn",
            "context_status": "metadata_only",
            "error_cell_index": "5",
            "prompt_context": {},
            "legacy_traceback_hint": None
        },
        {
            "notebook_execution_id": 15,
            "error_type": "ImportError",
            "error_message": "libxcb.so.1: cannot open shared object file: No such file or directory",
            "original_subtype": "system_library",
            "refined_subtype": "system_library",
            "scope_status": "excluded",
            "exclusion_reason": "requires system library, outside pip-only scope",
            "split": "excluded",
            "confidence": "high",
            "root_cause_hint": "system_level_dependency",
            "failing_module": "libxcb.so.1",
            "context_status": "metadata_only",
            "error_cell_index": "8",
            "prompt_context": {},
            "legacy_traceback_hint": None
        }
    ]

    input_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def fake_call_ollama(model, prompt, generation_config):
        return json.dumps(valid_response()), {"prompt_eval_count": 10, "eval_count": 20}

    monkeypatch.setattr(run_llm_explainer, "call_ollama", fake_call_ollama)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_llm_explainer.py",
            "--input", str(input_path),
            "--output", str(output_path),
            "--limit", "10",
            "--overwrite"
        ]
    )

    run_llm_explainer.main()

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    results = [json.loads(line) for line in lines]
    statuses = {r["explanation_result"]["status"] for r in results}
    assert statuses == {"success"}

    excluded_result = next(r for r in results if r["input"]["notebook_execution_id"] == 15)
    assert excluded_result["input"]["scope_status"] == "excluded"
    assert excluded_result["input"]["exclusion_reason"] == "requires system library, outside pip-only scope"
    assert excluded_result["explanation_result"]["status"] == "success"


def test_main_writes_render_failed_row(monkeypatch, tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    bad_template = tmp_path / "bad_template.txt"

    row = {
        "notebook_execution_id": 99,
        "error_type": "ModuleNotFoundError",
        "error_message": "No module named 'missing'",
        "original_subtype": "missing_package",
        "refined_subtype": "missing_package",
        "confidence": "high",
        "root_cause_hint": "direct_missing_package",
        "failing_module": "missing",
        "context_status": "metadata_only",
        "error_cell_index": "2",
        "prompt_context": {},
        "legacy_traceback_hint": None
    }

    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    bad_template.write_text("bad={{unknown_field}}", encoding="utf-8")

    cfg = config()
    cfg["output"] = {
        "schema": str(ROOT / "schemas" / "explanation.schema.json"),
        "path": str(output_path)
    }
    cfg["prompt"]["template"] = "does_not_matter"

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "runtime: ollama\n"
        "models:\n"
        "  primary: gemma2:9b\n"
        "  fallback: null\n"
        "generation:\n"
        "  temperature: 0.1\n"
        "  top_p: 0.9\n"
        "  max_tokens: 700\n"
        "  timeout_seconds: 120\n"
        "retry:\n"
        "  max_retries: 1\n"
        "  retry_on:\n"
        "    - invalid_json\n"
        "    - schema_validation_error\n"
        "    - timeout\n"
        "    - model_unavailable\n"
        "prompt:\n"
        "  strategy: few_shot\n"
        "  template: does_not_matter\n"
        "  version: i3_prompt_v1\n"
        "output:\n"
        f"  path: {output_path}\n"
        f"  schema: {ROOT / 'schemas' / 'explanation.schema.json'}\n",
        encoding="utf-8"
    )

    monkeypatch.setattr(
        run_llm_explainer,
        "iter_rendered_prompts",
        lambda input_path_arg, template_path_arg, limit=None: iter([
            {
                "index": 0,
                "record": row,
                "prompt": None,
                "error": "KeyError: Unknown template placeholder: unknown_field"
            }
        ])
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_llm_explainer.py",
            "--config", str(config_path),
            "--input", str(input_path),
            "--output", str(output_path),
            "--overwrite"
        ]
    )

    run_llm_explainer.main()

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    logged = json.loads(lines[0])
    assert logged["input"]["notebook_execution_id"] == 99
    assert logged["explanation_result"]["status"] == "render_failed"
    assert logged["explanation_result"]["failure_category"] == "render_failed"
    assert "unknown_field" in logged["explanation_result"]["error"]
