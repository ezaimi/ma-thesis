import json
import socket
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pypi_retriever
import rag_repair_agent
from pypi_retriever import load_rag_repair_config
from rag_repair_agent import (
    build_argv,
    build_command_display,
    check_eligibility,
    extract_wrong_version_signature,
    run_repair_agent,
)


@pytest.fixture(autouse=True)
def _reset_pypi_cache():
    pypi_retriever.clear_pypi_cache()
    yield
    pypi_retriever.clear_pypi_cache()


@pytest.fixture
def config():
    return load_rag_repair_config()


class FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def pypi_files_response(distribution: str, versions, requires_python=">=3.10"):
    return FakeResponse(json.dumps({
        "files": [
            {"filename": f"{distribution}-{v}.tar.gz", "yanked": False, "requires-python": requires_python}
            for v in versions
        ]
    }).encode("utf-8"))


def mock_pypi(monkeypatch, distribution: str, versions, requires_python=">=3.10"):
    def fake_urlopen(request, timeout=None):
        return pypi_files_response(distribution, versions, requires_python)

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)


def mock_pypi_404(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", None, None)

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)


def ollama_response(action, install_name, version, rationale="grounded rationale"):
    return json.dumps({
        "action": action, "install_name": install_name, "version": version, "rationale": rationale,
    })


def sklearn_record(scope_status="usable", exclusion_reason=""):
    return {
        "notebook_execution_id": 8,
        "error_type": "ModuleNotFoundError",
        "error_message": "No module named 'sklearn'",
        "original_subtype": "missing_package",
        "refined_subtype": "missing_package",
        "scope_status": scope_status,
        "exclusion_reason": exclusion_reason,
        "split": "dev",
        "failing_module": "sklearn",
        "root_cause_hint": "import_distribution_name_mismatch",
        "context_status": "metadata_only",
        "prompt_context": {},
    }


def cumtrapz_record(scope_status="usable"):
    return {
        "notebook_execution_id": 174,
        "error_type": "ImportError",
        "error_message": (
            "cannot import name 'cumtrapz' from 'scipy.integrate' "
            "(/tmp/.local/lib/python3.10/site-packages/scipy/integrate/__init__.py)"
        ),
        "original_subtype": "wrong_version",
        "refined_subtype": "wrong_version",
        "scope_status": scope_status,
        "exclusion_reason": "",
        "split": "dev",
        "failing_module": "scipy",
        "root_cause_hint": "version_or_api_incompatibility",
        "context_status": "metadata_only",
        "prompt_context": {},
    }


def isshape_record():
    return {
        "notebook_execution_id": 157,
        "error_type": "ImportError",
        "error_message": (
            "cannot import name 'isshape' from 'scipy.sparse.sputils' "
            "(/tmp/.local/lib/python3.10/site-packages/scipy/sparse/sputils.py)"
        ),
        "original_subtype": "wrong_version",
        "refined_subtype": "wrong_version",
        "scope_status": "usable",
        "exclusion_reason": "",
        "split": "dev",
        "failing_module": "scipy",
        "root_cause_hint": "version_or_api_incompatibility",
        "context_status": "metadata_only",
        "prompt_context": {},
    }


def numpy_warning_record():
    return {
        "notebook_execution_id": 188,
        "error_type": "ImportError",
        "error_message": (
            "cannot import name 'VisibleDeprecationWarning' from 'numpy' "
            "(/tmp/.local/lib/python3.10/site-packages/numpy/__init__.py)"
        ),
        "original_subtype": "wrong_version",
        "refined_subtype": "wrong_version",
        "scope_status": "usable",
        "exclusion_reason": "",
        "split": "dev",
        "failing_module": "numpy",
        "root_cause_hint": "version_or_api_incompatibility",
        "context_status": "metadata_only",
        "prompt_context": {},
    }


def system_library_record():
    return {
        "notebook_execution_id": 15,
        "error_type": "ImportError",
        "error_message": "libxcb.so.1: cannot open shared object file: No such file or directory",
        "original_subtype": "system_library",
        "refined_subtype": "system_library",
        "scope_status": "excluded",
        "exclusion_reason": "requires system library, outside pip-only scope",
        "split": "excluded",
        "failing_module": "libxcb.so.1",
        "root_cause_hint": "system_level_dependency",
        "context_status": "metadata_only",
        "prompt_context": {},
    }


def mapping_unknown_record():
    return {
        "notebook_execution_id": 50,
        "error_type": "ModuleNotFoundError",
        "error_message": "No module named 'dms_variants'",
        "original_subtype": "missing_package",
        "refined_subtype": "missing_package",
        "scope_status": "usable",
        "exclusion_reason": "",
        "split": "dev",
        "failing_module": "dms_variants",
        "root_cause_hint": "insufficient_context",
        "context_status": "metadata_only",
        "prompt_context": {},
    }


# --- 1. eligibility gate ------------------------------------------------------

def test_check_eligibility_usable():
    decision, reason = check_eligibility(sklearn_record())
    assert decision == "usable"
    assert reason is None


def test_check_eligibility_excluded_preserves_reason():
    decision, reason = check_eligibility(system_library_record())
    assert decision == "excluded"
    assert reason == "requires system library, outside pip-only scope"


def test_check_eligibility_missing_scope_status_is_invalid():
    record = sklearn_record()
    del record["scope_status"]
    decision, reason = check_eligibility(record)
    assert decision == "invalid"


def test_check_eligibility_unrecognized_scope_status_is_invalid():
    record = sklearn_record(scope_status="something_unexpected")
    decision, reason = check_eligibility(record)
    assert decision == "invalid"


def test_excluded_row_causes_zero_retrieval_and_llm_calls(monkeypatch, config):
    def fail_if_network(*a, **k):
        raise AssertionError("must not query PyPI for an excluded row")

    def fail_if_llm(*a, **k):
        raise AssertionError("must not call the LLM for an excluded row")

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fail_if_network)
    monkeypatch.setattr(rag_repair_agent, "call_ollama", fail_if_llm)

    result = run_repair_agent(system_library_record(), config)

    assert result["status"] == "abstained"
    assert result["final_action"] == "none"
    assert result["eligibility"]["decision"] == "excluded"
    assert result["eligibility"]["exclusion_reason"] == "requires system library, outside pip-only scope"
    assert result["retrieval_result"] is None
    assert result["llm"] is None


def test_invalid_scope_status_causes_zero_retrieval_and_llm_calls(monkeypatch, config):
    def fail_if_network(*a, **k):
        raise AssertionError("must not query PyPI when scope_status is missing/invalid")

    def fail_if_llm(*a, **k):
        raise AssertionError("must not call the LLM when scope_status is missing/invalid")

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fail_if_network)
    monkeypatch.setattr(rag_repair_agent, "call_ollama", fail_if_llm)

    record = sklearn_record()
    del record["scope_status"]

    result = run_repair_agent(record, config)

    assert result["status"] == "abstained"
    assert result["eligibility"]["decision"] == "invalid"
    assert result["retrieval_result"] is None


def test_usable_missing_package_row_proceeds_to_retrieval(monkeypatch, config):
    mock_pypi(monkeypatch, "scikit-learn", ["1.7.2"])
    monkeypatch.setattr(
        rag_repair_agent, "call_ollama",
        lambda **kwargs: (ollama_response("install", "scikit-learn", None), {}),
    )

    result = run_repair_agent(sklearn_record(), config)

    assert result["eligibility"]["decision"] == "usable"
    assert result["retrieval_result"] is not None
    assert result["retrieval_result"]["status"] == "resolved"
    assert result["llm"] is not None


def test_usable_wrong_version_row_proceeds_to_retrieval(monkeypatch, config):
    mock_pypi(monkeypatch, "scipy", ["1.13.0", "1.13.1", "1.14.0"])
    monkeypatch.setattr(
        rag_repair_agent, "call_ollama",
        lambda **kwargs: (ollama_response("pin_version", "scipy", "1.13.1"), {}),
    )

    result = run_repair_agent(cumtrapz_record(), config)

    assert result["eligibility"]["decision"] == "usable"
    assert result["retrieval_result"] is not None
    assert result["retrieval_result"]["subtype"] == "wrong_version"
    assert result["llm"] is not None


# --- 2. deterministic signature extraction -----------------------------------

def test_extract_signature_cumtrapz():
    module_path, symbol = extract_wrong_version_signature(
        "cannot import name 'cumtrapz' from 'scipy.integrate' (/some/path/__init__.py)"
    )
    assert module_path == "scipy.integrate"
    assert symbol == "cumtrapz"


def test_extract_signature_isshape():
    module_path, symbol = extract_wrong_version_signature(
        "cannot import name 'isshape' from 'scipy.sparse.sputils' (/some/path/sputils.py)"
    )
    assert module_path == "scipy.sparse.sputils"
    assert symbol == "isshape"


def test_extract_signature_visible_deprecation_warning():
    module_path, symbol = extract_wrong_version_signature(
        "cannot import name 'VisibleDeprecationWarning' from 'numpy' (/some/path/__init__.py)"
    )
    assert module_path == "numpy"
    assert symbol == "VisibleDeprecationWarning"


def test_extract_signature_malformed_message_returns_none_none():
    module_path, symbol = extract_wrong_version_signature("some unrelated error text")
    assert (module_path, symbol) == (None, None)


def test_extract_signature_empty_message_returns_none_none():
    assert extract_wrong_version_signature("") == (None, None)
    assert extract_wrong_version_signature(None) == (None, None)


def test_wrong_version_extraction_failure_abstains_without_pypi_or_llm(monkeypatch, config):
    def fail_if_network(*a, **k):
        raise AssertionError("must not query PyPI when extraction fails")

    def fail_if_llm(*a, **k):
        raise AssertionError("must not call the LLM when extraction fails")

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fail_if_network)
    monkeypatch.setattr(rag_repair_agent, "call_ollama", fail_if_llm)

    record = cumtrapz_record()
    record["error_message"] = "some unrelated error text that does not match the pattern"

    result = run_repair_agent(record, config)

    assert result["status"] == "abstained"
    assert result["extracted_signature"]["status"] == "failed"
    assert result["retrieval_result"] is None


# --- retrieval short-circuits: mapping_unknown / empty candidates -----------

def test_mapping_unknown_abstains_without_llm_call(monkeypatch, config):
    def fail_if_network(*a, **k):
        raise AssertionError("mapping_unknown must not trigger a PyPI request")

    def fail_if_llm(*a, **k):
        raise AssertionError("must not call the LLM when the import name has no verified mapping")

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fail_if_network)
    monkeypatch.setattr(rag_repair_agent, "call_ollama", fail_if_llm)

    result = run_repair_agent(mapping_unknown_record(), config)

    assert result["status"] == "abstained"
    assert result["retrieval_result"]["status"] == "mapping_unknown"
    assert result["llm"] is None


def test_empty_candidate_set_abstains_without_llm_call(monkeypatch, config):
    mock_pypi(monkeypatch, "scikit-learn", ["1.0.0"], requires_python="<3.7")  # incompatible with 3.10

    def fail_if_llm(*a, **k):
        raise AssertionError("must not call the LLM with an empty candidate set")

    monkeypatch.setattr(rag_repair_agent, "call_ollama", fail_if_llm)

    result = run_repair_agent(sklearn_record(), config)

    assert result["status"] == "abstained"
    assert result["retrieval_result"]["status"] == "no_compatible_release"
    assert result["llm"] is None


# --- 6. retry behavior ---------------------------------------------------------

def test_retry_after_malformed_json_then_valid_response(monkeypatch, config):
    mock_pypi(monkeypatch, "scikit-learn", ["1.7.2"])
    calls = {"count": 0}

    def fake_call_ollama(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return "not json", {}
        return ollama_response("install", "scikit-learn", None), {}

    monkeypatch.setattr(rag_repair_agent, "call_ollama", fake_call_ollama)

    result = run_repair_agent(sklearn_record(), config)

    assert calls["count"] == 2
    assert result["attempts"] == 2
    assert result["status"] == "success"
    assert result["final_action"] == "install"


def test_two_invalid_responses_produce_abstention(monkeypatch, config):
    mock_pypi(monkeypatch, "scikit-learn", ["1.7.2"])
    calls = {"count": 0}

    def fake_call_ollama(**kwargs):
        calls["count"] += 1
        return "not json", {}

    monkeypatch.setattr(rag_repair_agent, "call_ollama", fake_call_ollama)

    result = run_repair_agent(sklearn_record(), config)

    assert calls["count"] == 2
    assert result["status"] == "abstained"
    assert result["final_action"] == "none"
    assert result["argv"] is None


def test_grounding_invalid_response_retries_then_abstains(monkeypatch, config):
    mock_pypi(monkeypatch, "scikit-learn", ["1.7.2"])
    calls = {"count": 0}

    def fake_call_ollama(**kwargs):
        calls["count"] += 1
        # invents a distribution name not equal to the resolved one, both times
        return ollama_response("install", "totally-invented-package", None), {}

    monkeypatch.setattr(rag_repair_agent, "call_ollama", fake_call_ollama)

    result = run_repair_agent(sklearn_record(), config)

    assert calls["count"] == 2
    assert result["status"] == "abstained"
    assert result["grounding_validation"]["valid"] is False


def test_timeout_is_retried(monkeypatch, config):
    mock_pypi(monkeypatch, "scikit-learn", ["1.7.2"])
    calls = {"count": 0}

    def fake_call_ollama(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise socket.timeout("slow model")
        return ollama_response("install", "scikit-learn", None), {}

    monkeypatch.setattr(rag_repair_agent, "call_ollama", fake_call_ollama)
    monkeypatch.setattr(rag_repair_agent.time, "sleep", lambda *_: None)

    result = run_repair_agent(sklearn_record(), config)

    assert calls["count"] == 2
    assert result["status"] == "success"


def test_never_more_than_two_total_llm_calls(monkeypatch, config):
    mock_pypi(monkeypatch, "scikit-learn", ["1.7.2"])
    calls = {"count": 0}

    def fake_call_ollama(**kwargs):
        calls["count"] += 1
        return "not json", {}

    monkeypatch.setattr(rag_repair_agent, "call_ollama", fake_call_ollama)

    run_repair_agent(sklearn_record(), config)

    assert calls["count"] <= 2


# --- 11. deterministic argv construction --------------------------------------

def test_build_argv_install():
    assert build_argv("install", "scikit-learn", None) == ["python", "-m", "pip", "install", "scikit-learn"]


def test_build_argv_pin_version():
    assert build_argv("pin_version", "scipy", "1.13.1") == ["python", "-m", "pip", "install", "scipy==1.13.1"]


def test_build_argv_none_is_null():
    assert build_argv("none", None, None) is None


def test_build_command_display_is_non_executable_string():
    argv = build_argv("install", "scikit-learn", None)
    display = build_command_display(argv)
    assert display == "python -m pip install scikit-learn"
    assert isinstance(display, str)


def test_module_never_calls_subprocess_or_executes_anything():
    """Docstrings are allowed to *mention* subprocess while explaining that
    it's never used (and do); the module must never import it or call it."""
    source = Path(rag_repair_agent.__file__).read_text(encoding="utf-8")
    assert "import subprocess" not in source
    assert "subprocess.run" not in source
    assert "subprocess.call" not in source
    assert "subprocess.Popen" not in source
    assert "os.system" not in source
    assert "shell=True" not in source


# --- 5. end-to-end mocked component -------------------------------------------

def test_end_to_end_sklearn_missing_package_success(monkeypatch, config):
    mock_pypi(monkeypatch, "scikit-learn", ["1.7.2", "1.7.1"])
    monkeypatch.setattr(
        rag_repair_agent, "call_ollama",
        lambda **kwargs: (ollama_response("install", "scikit-learn", None), {}),
    )

    result = run_repair_agent(sklearn_record(), config)

    assert result["status"] == "success"
    assert result["final_action"] == "install"
    assert result["final_install_name"] == "scikit-learn"
    assert result["final_version"] is None
    assert result["argv"] == ["python", "-m", "pip", "install", "scikit-learn"]


def test_end_to_end_scipy_cumtrapz_wrong_version_success(monkeypatch, config):
    mock_pypi(monkeypatch, "scipy", ["1.13.0", "1.13.1", "1.14.0", "1.14.1"])
    monkeypatch.setattr(
        rag_repair_agent, "call_ollama",
        lambda **kwargs: (ollama_response("pin_version", "scipy", "1.13.1"), {}),
    )

    result = run_repair_agent(cumtrapz_record(), config)

    assert result["status"] == "success"
    assert result["final_action"] == "pin_version"
    assert result["final_install_name"] == "scipy"
    assert result["final_version"] == "1.13.1"
    assert result["argv"] == ["python", "-m", "pip", "install", "scipy==1.13.1"]
    assert result["retrieval_result"]["compatibility_evidence"]["compatible_specifier"] == "<1.14.0"


def test_end_to_end_scipy_isshape_wrong_version_success(monkeypatch, config):
    mock_pypi(monkeypatch, "scipy", ["1.13.1", "1.14.0"])
    monkeypatch.setattr(
        rag_repair_agent, "call_ollama",
        lambda **kwargs: (ollama_response("pin_version", "scipy", "1.13.1"), {}),
    )

    result = run_repair_agent(isshape_record(), config)

    assert result["status"] == "success"
    assert result["final_version"] == "1.13.1"


def test_end_to_end_numpy_visible_deprecation_warning_success(monkeypatch, config):
    mock_pypi(monkeypatch, "numpy", ["1.26.4", "2.0.0"])
    monkeypatch.setattr(
        rag_repair_agent, "call_ollama",
        lambda **kwargs: (ollama_response("pin_version", "numpy", "1.26.4"), {}),
    )

    result = run_repair_agent(numpy_warning_record(), config)

    assert result["status"] == "success"
    assert result["final_install_name"] == "numpy"
    assert result["final_version"] == "1.26.4"
    assert result["retrieval_result"]["compatibility_evidence"]["compatible_specifier"] == "<2.0.0"


def test_end_to_end_mapping_unknown_safe_abstention(monkeypatch, config):
    def fail_if_llm(*a, **k):
        raise AssertionError("must not call the LLM for mapping_unknown")

    monkeypatch.setattr(rag_repair_agent, "call_ollama", fail_if_llm)

    result = run_repair_agent(mapping_unknown_record(), config)

    assert result["status"] == "abstained"
    assert result["final_action"] == "none"
    assert result["argv"] is None


def test_end_to_end_empty_candidate_set_safe_abstention(monkeypatch, config):
    mock_pypi_404(monkeypatch)

    def fail_if_llm(*a, **k):
        raise AssertionError("must not call the LLM for package_not_found")

    monkeypatch.setattr(rag_repair_agent, "call_ollama", fail_if_llm)

    result = run_repair_agent(sklearn_record(), config)

    assert result["status"] == "abstained"
    assert result["retrieval_result"]["status"] == "package_not_found"
    assert result["argv"] is None


# --- record/line correspondence ------------------------------------------------

# --- CLI / batch runner -------------------------------------------------------

def test_cli_writes_one_line_per_record_with_mocked_pypi_and_ollama(monkeypatch, tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    rows = [sklearn_record(), system_library_record()]
    input_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    mock_pypi(monkeypatch, "scikit-learn", ["1.7.2"])
    monkeypatch.setattr(
        rag_repair_agent, "call_ollama",
        lambda **kwargs: (ollama_response("install", "scikit-learn", None), {}),
    )
    monkeypatch.setattr(
        sys, "argv",
        [
            "rag_repair_agent.py",
            "--input", str(input_path),
            "--output", str(output_path),
            "--limit", "10",
            "--overwrite",
        ],
    )

    rag_repair_agent.main()

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    results = [json.loads(line) for line in lines]
    assert results[0]["status"] == "success"
    assert results[1]["status"] == "abstained"


def test_cli_respects_limit_and_start_index(monkeypatch, tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    rows = [sklearn_record(), sklearn_record(), sklearn_record()]
    input_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    mock_pypi(monkeypatch, "scikit-learn", ["1.7.2"])
    monkeypatch.setattr(
        rag_repair_agent, "call_ollama",
        lambda **kwargs: (ollama_response("install", "scikit-learn", None), {}),
    )
    monkeypatch.setattr(
        sys, "argv",
        [
            "rag_repair_agent.py",
            "--input", str(input_path),
            "--output", str(output_path),
            "--start-index", "1",
            "--limit", "1",
            "--overwrite",
        ],
    )

    rag_repair_agent.main()

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["index"] == 1


def test_cli_defaults_to_append_not_overwrite(monkeypatch, tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    output_path.write_text('{"pre-existing": true}\n', encoding="utf-8")

    input_path.write_text(json.dumps(sklearn_record()), encoding="utf-8")

    mock_pypi(monkeypatch, "scikit-learn", ["1.7.2"])
    monkeypatch.setattr(
        rag_repair_agent, "call_ollama",
        lambda **kwargs: (ollama_response("install", "scikit-learn", None), {}),
    )
    monkeypatch.setattr(
        sys, "argv",
        ["rag_repair_agent.py", "--input", str(input_path), "--output", str(output_path), "--limit", "1"],
    )

    rag_repair_agent.main()

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"pre-existing": True}


def test_one_result_dict_per_record_regardless_of_llm_attempts(monkeypatch, config):
    mock_pypi(monkeypatch, "scikit-learn", ["1.7.2"])
    calls = {"count": 0}

    def fake_call_ollama(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return "not json", {}
        return ollama_response("install", "scikit-learn", None), {}

    monkeypatch.setattr(rag_repair_agent, "call_ollama", fake_call_ollama)

    result = run_repair_agent(sklearn_record(), config)

    assert isinstance(result, dict)
    assert result["notebook_execution_id"] == 8
    assert result["attempts"] == 2
