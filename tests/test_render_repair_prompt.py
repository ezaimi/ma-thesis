import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_repair_prompt import build_repair_template_values, render_repair_prompt


TEMPLATE_PATH = ROOT / "prompts" / "dependency_repair_v1.txt"


def sample_record():
    return {
        "notebook_execution_id": 174,
        "error_type": "ImportError",
        "error_message": "cannot import name 'cumtrapz' from 'scipy.integrate'",
        "refined_subtype": "wrong_version",
        "root_cause_hint": "version_or_api_incompatibility",
        "failing_module": "scipy",
        "context_status": "metadata_only",
        "prompt_context": {
            "failing_cell_source": None,
            "import_cells": [],
            "surrounding_cells": [],
        },
    }


def wrong_version_retrieval_result():
    return {
        "distribution_name": "scipy",
        "python_version": "3.10",
        "candidate_versions": [
            {"version": "1.13.1", "python_compatibility": "compatible"},
            {"version": "1.13.0", "python_compatibility": "compatible"},
        ],
        "compatibility_evidence": {
            "status": "resolved",
            "compatible_specifier": "<1.14.0",
            "evidence": {
                "summary": "SciPy 1.14.0 removed cumtrapz.",
                "source_url": "https://docs.scipy.org/doc/scipy/release/1.14.0-notes.html",
            },
        },
        "warnings": [],
    }


def missing_package_retrieval_result():
    return {
        "distribution_name": "scikit-learn",
        "python_version": "3.10",
        "candidate_versions": [
            {"version": "1.7.2", "python_compatibility": "compatible"},
        ],
        "compatibility_evidence": None,
        "warnings": [],
    }


def test_template_contains_grounded_candidate_versions():
    values = build_repair_template_values(sample_record(), wrong_version_retrieval_result(), "wrong_version")

    assert "1.13.1" in values["candidate_versions"]
    assert "1.13.0" in values["candidate_versions"]


def test_template_contains_compatibility_evidence_for_wrong_version():
    values = build_repair_template_values(sample_record(), wrong_version_retrieval_result(), "wrong_version")

    assert values["compatibility_constraint"] == "<1.14.0"
    assert "cumtrapz" in values["compatibility_evidence_summary"]
    assert "docs.scipy.org" in values["compatibility_evidence_summary"]


def test_missing_values_render_as_not_available():
    values = build_repair_template_values(sample_record(), missing_package_retrieval_result(), "missing_package")

    assert values["compatibility_constraint"] == "Not available"
    assert values["compatibility_evidence_summary"] == "Not available"


def test_no_candidates_renders_as_not_available():
    empty_result = {
        "distribution_name": None,
        "python_version": "3.10",
        "candidate_versions": [],
        "compatibility_evidence": None,
        "warnings": [],
    }
    values = build_repair_template_values(sample_record(), empty_result, "missing_package")

    assert values["candidate_versions"] == "Not available"
    assert values["distribution_name"] == "Not available"


def test_rendered_prompt_never_asks_for_a_command_field():
    """The prompt may (and does) explicitly forbid a "command" field, but no
    example JSON output block may show the model actually producing one."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    prompt = render_repair_prompt(sample_record(), wrong_version_retrieval_result(), "wrong_version", template)

    assert "shell command" in prompt.lower() or "pip command" in prompt.lower()

    # every example "Output:" JSON block must be exactly the 4-field shape
    for block in prompt.split("Output:\n")[1:]:
        json_text = block.split("\n\n")[0]
        example = json.loads(json_text)
        assert set(example.keys()) == {"action", "install_name", "version", "rationale"}


def test_rendered_prompt_contains_only_supplied_candidates_not_arbitrary_ones():
    """The rendered prompt must not contain any version string beyond what
    was actually supplied in candidate_versions - guards against a future
    template change accidentally leaking unfiltered PyPI data."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    prompt = render_repair_prompt(sample_record(), wrong_version_retrieval_result(), "wrong_version", template)

    # 1.14.0 was excluded by the compatibility intersection - it must never
    # appear in the rendered prompt's own candidate-version listing.
    candidate_section = prompt.split("candidate_versions:")[1].split("compatibility constraint:")[0]
    assert "1.14.0" not in candidate_section


def test_render_repair_prompt_fills_every_placeholder():
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    prompt = render_repair_prompt(sample_record(), wrong_version_retrieval_result(), "wrong_version", template)

    assert "{{" not in prompt
