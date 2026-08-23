#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from pypi_retriever import normalize_distribution_name


DEFAULT_EVIDENCE_PATH = Path(__file__).resolve().parent.parent / "config" / "api_compatibility_evidence.yaml"

REQUIRED_ENTRY_FIELDS = {"distribution_name", "module_path", "symbol", "compatible_specifier", "evidence"}
REQUIRED_EVIDENCE_FIELDS = {"source_type", "source_url", "summary", "verified_at"}


def load_compatibility_evidence(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the curated pattern registry as {"<module_path>.<symbol>": entry}.

    Never makes a network request; reads the local YAML file only.
    """
    evidence_path = Path(path) if path else DEFAULT_EVIDENCE_PATH
    with open(evidence_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("patterns", {}) or {}


def _pattern_key(module_path: str, symbol: str) -> str:
    return f"{module_path}.{symbol}"


def _validate_entry(entry: Any, module_path: str, symbol: str, distribution_name: str) -> Optional[str]:
    """Return an error string if `entry` is not a safe, usable evidence
    record for (distribution_name, module_path, symbol); None if it's valid."""
    if not isinstance(entry, dict):
        return "registry entry is not a mapping"

    missing = REQUIRED_ENTRY_FIELDS - entry.keys()
    if missing:
        return f"registry entry missing required field(s): {sorted(missing)}"

    evidence_block = entry.get("evidence")
    if not isinstance(evidence_block, dict):
        return "registry entry's 'evidence' field is not a mapping"

    missing_evidence = REQUIRED_EVIDENCE_FIELDS - evidence_block.keys()
    if missing_evidence:
        return f"evidence block missing required field(s): {sorted(missing_evidence)}"

    if entry.get("module_path") != module_path or entry.get("symbol") != symbol:
        return "registry entry's module_path/symbol does not match its own key"

    if normalize_distribution_name(entry.get("distribution_name", "")) != normalize_distribution_name(distribution_name):
        return "registry entry's distribution_name does not match the requested distribution"

    source_url = evidence_block.get("source_url", "")
    if not isinstance(source_url, str) or not source_url.startswith("https://"):
        return "evidence source_url must be an https:// URL"

    try:
        SpecifierSet(entry["compatible_specifier"])
    except InvalidSpecifier as exc:
        return f"invalid compatible_specifier: {exc}"

    return None


def lookup_compatibility_evidence(
    distribution_name: str,
    module_path: str,
    symbol: str,
    patterns: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Look up curated API-compatibility evidence for one import pattern.

    module_path and symbol are matched exactly and case-sensitively (they are
    Python identifiers). distribution_name is compared after PEP 503
    normalization only, since PyPI distribution names have no reliable
    case convention. Never makes a network request.
    """
    if patterns is None:
        patterns = load_compatibility_evidence()

    result: Dict[str, Any] = {
        "status": None,
        "distribution_name": distribution_name,
        "module_path": module_path,
        "symbol": symbol,
        "compatible_specifier": None,
        "incompatible_from": None,
        "evidence": None,
        "error": None,
    }

    entry = patterns.get(_pattern_key(module_path, symbol))

    if entry is None:
        result["status"] = "no_evidence"
        return result

    error = _validate_entry(entry, module_path, symbol, distribution_name)
    if error:
        result["status"] = "invalid_entry"
        result["error"] = error
        return result

    result["status"] = "resolved"
    result["compatible_specifier"] = entry["compatible_specifier"]
    result["incompatible_from"] = entry.get("incompatible_from")
    result["evidence"] = dict(entry["evidence"])
    return result


def filter_versions_by_compatibility_evidence(
    candidate_versions: List[Dict[str, Any]],
    evidence: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Return only the candidate versions the compatibility evidence proves
    still contain the required symbol.

    `candidate_versions` is the same shape as the L5 retrieval-result
    candidate list (each dict has at least a "version" key). `evidence` is
    the dict returned by lookup_compatibility_evidence(). Returns [] unless
    evidence["status"] == "resolved" -- no evidence and invalid evidence both
    mean no version can be safely claimed compatible. Never makes a network
    request and never constructs or runs a pip command.
    """
    if evidence.get("status") != "resolved":
        return []

    try:
        specifier = SpecifierSet(evidence["compatible_specifier"])
    except InvalidSpecifier:
        return []

    survivors = []
    for candidate in candidate_versions:
        raw_version = candidate.get("version")
        try:
            parsed_version = Version(raw_version)
        except (InvalidVersion, TypeError):
            print(
                f"[WARN] Skipping candidate with unparseable version: {raw_version!r}",
                file=sys.stderr,
            )
            continue

        if parsed_version in specifier:
            survivors.append(candidate)

    return survivors
