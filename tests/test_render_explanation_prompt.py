import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_explanation_prompt import build_template_values, render_prompt


def sample_record():
    return {
        "notebook_execution_id": 8,
        "error_type": "ModuleNotFoundError",
        "error_message": "No module named 'sklearn'",
        "original_subtype": "missing_package",
        "refined_subtype": "missing_package",
        "confidence": "high",
        "root_cause_hint": "import_distribution_name_mismatch",
        "failing_module": "sklearn",
        "context_status": "metadata_only",
        "error_cell_index": "5",
        "prompt_context": {
            "failing_cell_source": None,
            "import_cells": [],
            "surrounding_cells": [],
            "dependency_files": []
        },
        "legacy_traceback_hint": None
    }


def test_missing_context_renders_not_available():
    values = build_template_values(sample_record())

    assert values["failing_cell_source"] == "Not available"
    assert values["import_cells"] == "Not available"
    assert values["surrounding_cells"] == "Not available"
    assert values["dependency_files"] == "Not available"


def test_template_placeholders_are_replaced():
    template = "error_type={{error_type}}\nfailing_module={{failing_module}}"
    values = build_template_values(sample_record())

    rendered = render_prompt(template, values)

    assert "ModuleNotFoundError" in rendered
    assert "sklearn" in rendered
    assert "{{" not in rendered


def test_unknown_placeholder_raises_keyerror():
    values = build_template_values(sample_record())

    try:
        render_prompt("unknown={{unknown_field}}", values)
    except KeyError as e:
        assert "unknown_field" in str(e)
    else:
        raise AssertionError("Expected KeyError")


def test_single_pass_replacement_does_not_replace_injected_template_text():
    record = sample_record()
    record["prompt_context"]["failing_cell_source"] = "render({{root_cause_hint}})"
    values = build_template_values(record)

    rendered = render_prompt("cell={{failing_cell_source}}", values)

    assert "render({{root_cause_hint}})" in rendered

import json


def test_real_i2_jsonl_row_maps_confidence_to_classifier_confidence():
    path = ROOT / "data" / "context-classification" / "dependency_error_contexts.jsonl"

    with path.open(encoding="utf-8") as f:
        record = json.loads(next(f))

    values = build_template_values(record)

    assert values["classifier_confidence"] == record["confidence"]
    assert values["refined_subtype"] == record["refined_subtype"]
    assert values["failing_module"] == record["failing_module"]


def test_legacy_traceback_hint_dict_unwraps_raw_traceback():
    record = sample_record()
    record["legacy_traceback_hint"] = {
        "accepted": True,
        "raw_traceback": "Traceback text here"
    }

    values = build_template_values(record)

    assert values["legacy_traceback_hint"] == "Traceback text here"


def test_system_library_record_renders_through_the_real_explanation_template():
    """LLMExplainer must be able to render an explanation prompt for an
    excluded (system_library) subtype, since explanation scope covers all
    214 DEPENDENCY_ERROR rows and repair eligibility (scope_status) is a
    separate, downstream concern. See docs/architecture-note.md §7.1."""
    from render_explanation_prompt import render_record_prompt

    record = sample_record()
    record.update({
        "notebook_execution_id": 15,
        "error_type": "ImportError",
        "error_message": "libxcb.so.1: cannot open shared object file: No such file or directory",
        "original_subtype": "system_library",
        "refined_subtype": "system_library",
        "confidence": "high",
        "root_cause_hint": "system_level_dependency",
        "failing_module": "libxcb.so.1",
        "scope_status": "excluded",
        "exclusion_reason": "requires system library, outside pip-only scope",
        "split": "excluded",
    })

    template_path = ROOT / "prompts" / "dependency_explanation_v1.txt"
    template = template_path.read_text(encoding="utf-8")

    prompt = render_record_prompt(record, template)

    assert "{{" not in prompt
    assert "libxcb.so.1" in prompt
    assert "system_level_dependency" in prompt
    # The explainer must never be told to withhold an explanation because of
    # repair eligibility - scope_status is not one of the template's slots.
    assert "scope_status" not in prompt


def test_iter_rendered_prompts_preserves_record_on_render_failure(tmp_path):
    from render_explanation_prompt import iter_rendered_prompts

    input_path = tmp_path / "input.jsonl"
    template_path = tmp_path / "template.txt"

    record = sample_record()
    input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    template_path.write_text("unknown={{unknown_field}}", encoding="utf-8")

    items = list(iter_rendered_prompts(str(input_path), str(template_path)))

    assert len(items) == 1
    assert items[0]["prompt"] is None
    assert items[0]["record"]["notebook_execution_id"] == 8
    assert "Unknown template placeholder" in items[0]["error"]
