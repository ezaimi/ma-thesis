import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from repair_proposal_validator import (
    is_safe_token,
    parse_and_validate_schema,
    parse_model_response,
    validate_grounding,
)


SCHEMA_PATH = str(ROOT / "schemas" / "repair_proposal.schema.json")


def missing_package_retrieval_result(candidate_versions=None):
    return {
        "status": "resolved",
        "import_name": "sklearn",
        "subtype": "missing_package",
        "distribution_name": "scikit-learn",
        "candidate_versions": candidate_versions if candidate_versions is not None else [
            {"version": "1.7.2", "python_compatibility": "compatible", "yanked": False},
            {"version": "1.7.1", "python_compatibility": "compatible", "yanked": False},
        ],
        "compatibility_evidence": None,
    }


def wrong_version_retrieval_result(candidate_versions=None, compat_status="resolved"):
    return {
        "status": "resolved",
        "import_name": "scipy",
        "subtype": "wrong_version",
        "distribution_name": "scipy",
        "candidate_versions": candidate_versions if candidate_versions is not None else [
            {"version": "1.13.1", "python_compatibility": "compatible", "yanked": False},
            {"version": "1.13.0", "python_compatibility": "compatible", "yanked": False},
        ],
        "compatibility_evidence": {
            "status": compat_status,
            "compatible_specifier": "<1.14.0" if compat_status == "resolved" else None,
            "evidence": {"source_url": "https://docs.scipy.org/doc/scipy/release/1.14.0-notes.html"},
        },
    }


# --- schema/parser: valid proposals ------------------------------------------

def test_valid_install_proposal_passes():
    raw = '{"action": "install", "install_name": "scikit-learn", "version": null, "rationale": "grounded"}'
    valid, proposal, errors = parse_and_validate_schema(raw, SCHEMA_PATH)
    assert valid is True
    assert proposal["action"] == "install"
    assert errors == []


def test_valid_pin_version_proposal_passes():
    raw = '{"action": "pin_version", "install_name": "scipy", "version": "1.13.1", "rationale": "grounded"}'
    valid, proposal, errors = parse_and_validate_schema(raw, SCHEMA_PATH)
    assert valid is True


def test_valid_none_proposal_passes():
    raw = '{"action": "none", "install_name": null, "version": null, "rationale": "insufficient evidence"}'
    valid, proposal, errors = parse_and_validate_schema(raw, SCHEMA_PATH)
    assert valid is True


# --- schema/parser: malformed input ------------------------------------------

def test_malformed_json_is_rejected():
    valid, proposal, errors = parse_and_validate_schema("not json{{{", SCHEMA_PATH)
    assert valid is False
    assert proposal is None
    assert "invalid_json" in errors[0]


def test_json_fenced_in_markdown_is_extracted():
    raw = '```json\n{"action": "none", "install_name": null, "version": null, "rationale": "x"}\n```'
    valid, proposal, errors = parse_and_validate_schema(raw, SCHEMA_PATH)
    assert valid is True
    assert proposal["action"] == "none"


def test_json_embedded_in_prose_is_extracted():
    raw = 'Here is my answer: {"action": "none", "install_name": null, "version": null, "rationale": "x"} Hope this helps!'
    valid, proposal, errors = parse_and_validate_schema(raw, SCHEMA_PATH)
    assert valid is True


def test_braces_inside_quoted_strings_do_not_break_extraction():
    raw = 'prefix {"action": "none", "install_name": null, "version": null, "rationale": "use {curly} braces"} suffix'
    proposal = parse_model_response(raw)
    assert proposal["rationale"] == "use {curly} braces"


def test_missing_required_field_is_rejected():
    raw = '{"action": "install", "install_name": "scikit-learn", "version": null}'
    valid, proposal, errors = parse_and_validate_schema(raw, SCHEMA_PATH)
    assert valid is False
    assert any("rationale" in e for e in errors)


def test_extra_command_field_is_rejected():
    raw = '{"action": "install", "install_name": "scikit-learn", "version": null, "rationale": "x", "command": "pip install scikit-learn"}'
    valid, proposal, errors = parse_and_validate_schema(raw, SCHEMA_PATH)
    assert valid is False


def test_extra_pip_command_field_is_rejected():
    raw = '{"action": "none", "install_name": null, "version": null, "rationale": "x", "pip_command": "pip install x"}'
    valid, proposal, errors = parse_and_validate_schema(raw, SCHEMA_PATH)
    assert valid is False


def test_extra_package_spec_field_is_rejected():
    raw = '{"action": "none", "install_name": null, "version": null, "rationale": "x", "package_spec": "scikit-learn>=1.0"}'
    valid, proposal, errors = parse_and_validate_schema(raw, SCHEMA_PATH)
    assert valid is False


def test_invalid_action_is_rejected():
    raw = '{"action": "upgrade", "install_name": "scikit-learn", "version": null, "rationale": "x"}'
    valid, proposal, errors = parse_and_validate_schema(raw, SCHEMA_PATH)
    assert valid is False


def test_empty_rationale_is_rejected():
    raw = '{"action": "none", "install_name": null, "version": null, "rationale": ""}'
    valid, proposal, errors = parse_and_validate_schema(raw, SCHEMA_PATH)
    assert valid is False


def test_install_with_non_null_version_is_rejected():
    raw = '{"action": "install", "install_name": "scikit-learn", "version": "1.7.2", "rationale": "x"}'
    valid, proposal, errors = parse_and_validate_schema(raw, SCHEMA_PATH)
    assert valid is False


def test_none_with_non_null_install_name_is_rejected():
    raw = '{"action": "none", "install_name": "scikit-learn", "version": null, "rationale": "x"}'
    valid, proposal, errors = parse_and_validate_schema(raw, SCHEMA_PATH)
    assert valid is False


# --- is_safe_token ------------------------------------------------------------

def test_is_safe_token_accepts_normal_distribution_name():
    assert is_safe_token("scikit-learn") is True


def test_is_safe_token_rejects_leading_dash():
    assert is_safe_token("-rf") is False


def test_is_safe_token_rejects_whitespace():
    assert is_safe_token("scikit learn") is False


def test_is_safe_token_rejects_shell_metacharacters():
    assert is_safe_token("scikit-learn; rm -rf /") is False
    assert is_safe_token("$(whoami)") is False
    assert is_safe_token("scikit-learn|cat") is False


def test_is_safe_token_rejects_control_characters():
    assert is_safe_token("scikit-learn\n--user") is False
    assert is_safe_token("scikit\x00learn") is False


def test_is_safe_token_rejects_empty_and_non_string():
    assert is_safe_token("") is False
    assert is_safe_token(None) is False


# --- grounding validation: missing_package -----------------------------------

def test_grounding_accepts_exact_mapped_distribution_for_missing_package():
    proposal = {"action": "install", "install_name": "scikit-learn", "version": None, "rationale": "x"}
    errors = validate_grounding(proposal, missing_package_retrieval_result(), "missing_package")
    assert errors == []


def test_grounding_rejects_invented_distribution_for_missing_package():
    proposal = {"action": "install", "install_name": "totally-made-up-package", "version": None, "rationale": "x"}
    errors = validate_grounding(proposal, missing_package_retrieval_result(), "missing_package")
    assert errors


def test_grounding_rejects_missing_package_install_with_no_compatible_candidate():
    result = missing_package_retrieval_result(candidate_versions=[
        {"version": "0.1.0", "python_compatibility": "unknown", "yanked": False},
    ])
    proposal = {"action": "install", "install_name": "scikit-learn", "version": None, "rationale": "x"}
    errors = validate_grounding(proposal, result, "missing_package")
    assert errors


def test_grounding_rejects_wrong_action_for_missing_package_subtype():
    proposal = {"action": "pin_version", "install_name": "scikit-learn", "version": "1.7.2", "rationale": "x"}
    errors = validate_grounding(proposal, missing_package_retrieval_result(), "missing_package")
    assert errors


# --- grounding validation: wrong_version -------------------------------------

def test_grounding_accepts_exact_candidate_version_for_wrong_version():
    proposal = {"action": "pin_version", "install_name": "scipy", "version": "1.13.1", "rationale": "x"}
    errors = validate_grounding(proposal, wrong_version_retrieval_result(), "wrong_version")
    assert errors == []


def test_grounding_rejects_version_not_in_candidate_set():
    proposal = {"action": "pin_version", "install_name": "scipy", "version": "1.16.0", "rationale": "x"}
    errors = validate_grounding(proposal, wrong_version_retrieval_result(), "wrong_version")
    assert errors


def test_grounding_rejects_version_range_instead_of_exact_version():
    proposal = {"action": "pin_version", "install_name": "scipy", "version": "<1.14.0", "rationale": "x"}
    errors = validate_grounding(proposal, wrong_version_retrieval_result(), "wrong_version")
    assert errors


def test_grounding_rejects_wrong_action_for_wrong_version_subtype():
    proposal = {"action": "install", "install_name": "scipy", "version": None, "rationale": "x"}
    errors = validate_grounding(proposal, wrong_version_retrieval_result(), "wrong_version")
    assert errors


def test_grounding_rejects_pin_version_with_missing_compatibility_evidence():
    result = wrong_version_retrieval_result(compat_status="no_evidence")
    proposal = {"action": "pin_version", "install_name": "scipy", "version": "1.13.1", "rationale": "x"}
    errors = validate_grounding(proposal, result, "wrong_version")
    assert errors


def test_grounding_rejects_shell_injection_in_install_name():
    proposal = {"action": "install", "install_name": "scikit-learn; rm -rf /", "version": None, "rationale": "x"}
    errors = validate_grounding(proposal, missing_package_retrieval_result(), "missing_package")
    assert errors


def test_grounding_rejects_version_beginning_with_dash():
    proposal = {"action": "pin_version", "install_name": "scipy", "version": "--upgrade", "rationale": "x"}
    errors = validate_grounding(proposal, wrong_version_retrieval_result(), "wrong_version")
    assert errors


def test_grounding_accepts_valid_none_proposal():
    proposal = {"action": "none", "install_name": None, "version": None, "rationale": "insufficient evidence"}
    errors = validate_grounding(proposal, missing_package_retrieval_result(), "missing_package")
    assert errors == []


def test_grounding_rejects_none_action_with_non_null_fields():
    proposal = {"action": "none", "install_name": "scikit-learn", "version": None, "rationale": "x"}
    errors = validate_grounding(proposal, missing_package_retrieval_result(), "missing_package")
    assert errors
