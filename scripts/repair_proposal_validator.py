#!/usr/bin/env python3

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jsonschema import Draft202012Validator


# A deliberately narrow, allowlist-style safety check for install_name and
# version strings that survive grounding validation. In practice this is
# belt-and-suspenders: both fields are only ever accepted when they exactly
# match a value this process already produced from a trusted source
# (config/package_mapping.yaml's resolved distribution_name, or a candidate
# version string pypi_retriever.py itself built via packaging.version.Version).
# An LLM-invented value that isn't an exact match is rejected on that basis
# alone - this regex exists so the safety property does not depend solely on
# the exact-match check ever being right.
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def is_safe_token(value: Any) -> bool:
    """True only for a non-empty string starting with an alphanumeric
    character, containing only letters, digits, '.', '_', '-'. Rejects
    leading '-' (which argv parsers could mistake for a flag), shell
    metacharacters (;|&`$(){}<>), and any whitespace or control character."""
    if not isinstance(value, str) or not value:
        return False
    return bool(_SAFE_TOKEN_RE.match(value))


# --- defensive response parsing (mirrors scripts/explanation_validator.py) --

def extract_fenced_json(text: str) -> Optional[str]:
    """Extract JSON from a markdown code fence if present, even when the
    model adds text before or after the fence."""
    fence_start = text.find("```")
    if fence_start == -1:
        return None

    content_start = text.find("\n", fence_start)
    if content_start == -1:
        return None

    fence_end = text.find("```", content_start + 1)
    if fence_end == -1:
        return None

    return text[content_start + 1:fence_end].strip()


def extract_first_json_object(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from text, respecting quoted
    strings so braces inside strings do not break parsing."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]

    return None


def parse_model_response(raw_response: str) -> Dict[str, Any]:
    """Parse a model response that should contain one JSON object.

    Accepted defensively: pure JSON object, markdown-fenced JSON, text
    containing one balanced JSON object. Rejected: non-JSON text, JSON
    arrays/strings/numbers.
    """
    text = raw_response.strip()

    candidates = [
        text,
        extract_fenced_json(text),
        extract_first_json_object(text),
    ]

    last_error = None

    for candidate in candidates:
        if not candidate:
            continue

        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as e:
            last_error = e
            continue

        if not isinstance(parsed, dict):
            raise ValueError("non_object_json: model response must be a JSON object")

        return parsed

    if last_error is not None:
        raise ValueError(f"invalid_json: {last_error}")

    raise ValueError("invalid_json: no JSON object found in model response")


# --- schema (shape) validation -----------------------------------------------

def load_schema(schema_path: str) -> Dict[str, Any]:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_proposal_schema(proposal: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Shape validation only - action/install_name/version/rationale types,
    the action enum, and the action-dependent nullability rules. Does not
    check the proposal against retrieval evidence; see validate_grounding()."""
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(proposal), key=lambda e: e.path)

    return [
        f"{'/'.join(map(str, error.path)) or '<root>'}: {error.message}"
        for error in errors
    ]


def parse_and_validate_schema(
    raw_response: str,
    schema_path: str,
) -> Tuple[bool, Optional[Dict[str, Any]], List[str]]:
    schema = load_schema(schema_path)

    try:
        proposal = parse_model_response(raw_response)
    except ValueError as e:
        return False, None, [str(e)]

    validation_errors = validate_proposal_schema(proposal, schema)

    if validation_errors:
        return False, proposal, validation_errors

    return True, proposal, []


# --- deterministic grounding validation --------------------------------------

def validate_grounding(
    proposal: Dict[str, Any],
    retrieval_result: Dict[str, Any],
    subtype: str,
) -> List[str]:
    """Validate a schema-valid proposal against what was actually retrieved.

    Never trusts the proposal's own text: install_name and version are only
    accepted when they exactly match a value the deterministic retriever
    already produced (docs/rag-design.md §2.2, step 3). Returns a list of
    error strings; empty means the proposal is grounded and safe to build an
    argv from. Any single failure means the whole proposal is rejected -
    never repaired or substituted with a "safe" guess.
    """
    errors: List[str] = []

    action = proposal.get("action")
    install_name = proposal.get("install_name")
    version = proposal.get("version")

    distribution_name = retrieval_result.get("distribution_name")
    candidate_versions = retrieval_result.get("candidate_versions") or []
    candidate_by_version = {c.get("version"): c for c in candidate_versions}

    if action == "none":
        if install_name is not None:
            errors.append("action is 'none' but install_name is not null")
        if version is not None:
            errors.append("action is 'none' but version is not null")
        return errors

    if action == "install":
        if subtype != "missing_package":
            errors.append(f"action 'install' is not valid for subtype '{subtype}'")

        if version is not None:
            errors.append("action 'install' must have version: null")

        if not is_safe_token(install_name):
            errors.append(f"install_name {install_name!r} contains unsafe characters")
        elif install_name != distribution_name:
            errors.append(
                f"install_name {install_name!r} does not exactly match the resolved "
                f"distribution {distribution_name!r}"
            )
        elif distribution_name is None:
            errors.append("no distribution_name was resolved by the retriever")
        else:
            compatible_candidates = [
                c for c in candidate_versions if c.get("python_compatibility") == "compatible"
            ]
            if not compatible_candidates:
                errors.append(
                    "no candidate_versions entry proves the resolved distribution exists "
                    "with confirmed Python compatibility"
                )

        return errors

    if action == "pin_version":
        if subtype != "wrong_version":
            errors.append(f"action 'pin_version' is not valid for subtype '{subtype}'")

        if not is_safe_token(install_name):
            errors.append(f"install_name {install_name!r} contains unsafe characters")
        elif install_name != distribution_name:
            errors.append(
                f"install_name {install_name!r} does not exactly match the resolved "
                f"distribution {distribution_name!r}"
            )

        if not is_safe_token(version):
            errors.append(f"version {version!r} contains unsafe characters")
        elif version not in candidate_by_version:
            errors.append(
                f"version {version!r} is not one of the retrieved candidate versions "
                f"{sorted(candidate_by_version)}"
            )

        compatibility_evidence = retrieval_result.get("compatibility_evidence") or {}
        if compatibility_evidence.get("status") != "resolved":
            errors.append(
                "no resolved API-compatibility evidence backs this wrong_version proposal "
                f"(compatibility_evidence status: {compatibility_evidence.get('status')!r})"
            )

        return errors

    errors.append(f"unsupported action: {action!r}")
    return errors
