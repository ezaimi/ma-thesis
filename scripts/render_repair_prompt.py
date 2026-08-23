#!/usr/bin/env python3

import re
from typing import Any, Dict, List, Optional


NOT_AVAILABLE = "Not available"
PLACEHOLDER_RE = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")

CONTEXT_STATUS_NOTES = {
    "metadata_only": "only classifier metadata was available; no notebook source could be inspected.",
    "legacy_execution_hint": "a legacy traceback hint was available, not the current notebook source.",
    "remote_fetched": "notebook source was fetched from the repository at retrieval time.",
    "source_not_found": "notebook source could not be fetched; only metadata is available.",
}


def format_value(value: Any) -> str:
    if value is None:
        return NOT_AVAILABLE

    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else NOT_AVAILABLE

    if isinstance(value, (list, dict)):
        if not value:
            return NOT_AVAILABLE
        return str(value)

    return str(value)


def get_prompt_context(record: Dict[str, Any]) -> Dict[str, Any]:
    context = record.get("prompt_context")
    return context if isinstance(context, dict) else {}


def format_candidate_versions(candidate_versions: Optional[List[Dict[str, Any]]]) -> str:
    if not candidate_versions:
        return NOT_AVAILABLE

    lines = [
        f"- {c.get('version')} (python_compatibility: {c.get('python_compatibility')})"
        for c in candidate_versions
    ]
    return "\n".join(lines)


def format_compatibility_constraint(retrieval_result: Dict[str, Any]) -> str:
    evidence = retrieval_result.get("compatibility_evidence")
    if not isinstance(evidence, dict):
        return NOT_AVAILABLE
    return format_value(evidence.get("compatible_specifier"))


def format_compatibility_evidence_summary(retrieval_result: Dict[str, Any]) -> str:
    evidence = retrieval_result.get("compatibility_evidence")
    if not isinstance(evidence, dict):
        return NOT_AVAILABLE

    source = evidence.get("evidence")
    if not isinstance(source, dict):
        return NOT_AVAILABLE

    summary = source.get("summary")
    source_url = source.get("source_url")
    if not summary:
        return NOT_AVAILABLE

    if source_url:
        return f"{summary} (source: {source_url})"
    return summary


def format_warnings(retrieval_result: Dict[str, Any]) -> str:
    warnings = retrieval_result.get("warnings")
    if not warnings:
        return "none"
    return "; ".join(warnings)


def format_limitations(record: Dict[str, Any]) -> str:
    context_status = record.get("context_status")
    note = CONTEXT_STATUS_NOTES.get(context_status)
    if note:
        return f"context_status: {context_status} - {note}"
    if context_status:
        return f"context_status: {context_status}"
    return NOT_AVAILABLE


def build_repair_template_values(
    record: Dict[str, Any],
    retrieval_result: Dict[str, Any],
    subtype: str,
) -> Dict[str, str]:
    prompt_context = get_prompt_context(record)

    return {
        "notebook_execution_id": format_value(record.get("notebook_execution_id")),
        "error_type": format_value(record.get("error_type")),
        "error_message": format_value(record.get("error_message")),
        "subtype": format_value(subtype),
        "root_cause_hint": format_value(record.get("root_cause_hint")),
        "failing_module": format_value(record.get("failing_module")),
        "failing_cell_source": format_value(prompt_context.get("failing_cell_source")),
        "import_cells": format_value(prompt_context.get("import_cells")),
        "surrounding_cells": format_value(prompt_context.get("surrounding_cells")),
        "distribution_name": format_value(retrieval_result.get("distribution_name")),
        "python_version": format_value(retrieval_result.get("python_version")),
        "candidate_versions": format_candidate_versions(retrieval_result.get("candidate_versions")),
        "compatibility_constraint": format_compatibility_constraint(retrieval_result),
        "compatibility_evidence_summary": format_compatibility_evidence_summary(retrieval_result),
        "retrieval_warnings": format_warnings(retrieval_result),
        "retrieval_limitations": format_limitations(record),
    }


def render_prompt(template: str, values: Dict[str, str]) -> str:
    def replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError("Unknown template placeholder: {}".format(key))
        return values[key]

    return PLACEHOLDER_RE.sub(replace, template)


def render_repair_prompt(
    record: Dict[str, Any],
    retrieval_result: Dict[str, Any],
    subtype: str,
    template: str,
) -> str:
    values = build_repair_template_values(record, retrieval_result, subtype)
    return render_prompt(template, values)
