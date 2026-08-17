#!/usr/bin/env python3

import json
import re
import socket
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import InvalidVersion, Version


DEFAULT_PACKAGE_MAPPING_PATH = Path(__file__).resolve().parent.parent / "config" / "package_mapping.yaml"
DEFAULT_RAG_REPAIR_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "rag_repair.yaml"

PYPI_SIMPLE_BASE = "https://pypi.org/simple"
PYPI_SIMPLE_ACCEPT_HEADER = "application/vnd.pypi.simple.v1+json"
DEFAULT_FETCH_TIMEOUT_SECONDS = 10.0
MAX_CANDIDATE_VERSIONS = 5


def load_package_mapping(path: Optional[str] = None) -> Dict[str, str]:
    mapping_path = Path(path) if path else DEFAULT_PACKAGE_MAPPING_PATH
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)
    return mapping or {}


_PACKAGE_MAPPING = load_package_mapping()


def load_rag_repair_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load config/rag_repair.yaml. python_version is a fixed, repo-wide
    constant (the Docker execution environment always runs Python 3.10 -
    see the file's own header comment and docs/rag-design.md §4) - it is
    never looked up per notebook."""
    config_path = Path(path) if path else DEFAULT_RAG_REPAIR_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


_RAG_REPAIR_CONFIG = load_rag_repair_config()
DEFAULT_PYTHON_VERSION = _RAG_REPAIR_CONFIG.get("runtime", {}).get("python_version")


def resolve_distribution_name(import_name: str) -> Optional[str]:
    return _PACKAGE_MAPPING.get(import_name)


def normalize_distribution_name(distribution_name: str) -> str:
    return re.sub(r"[-_.]+", "-", distribution_name).lower()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- PyPI Simple API client -------------------------------------------------

def fetch_pypi_project(
    distribution_name: str,
    timeout: float = DEFAULT_FETCH_TIMEOUT_SECONDS,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Fetch one distribution's file listing from the PyPI JSON Simple API.

    `distribution_name` must already be PEP 503-normalized by the caller.
    Returns (status, data): status is "ok" (data is the parsed JSON dict),
    "package_not_found", "network_error", or "invalid_response" (data is
    None in the latter three cases). Distinguishes HTTP, connection,
    timeout, and parsing failures rather than collapsing them into one
    generic error, per day-1-rag-repair-agent-plan.md Step 4.
    """
    url = f"{PYPI_SIMPLE_BASE}/{distribution_name}/"
    request = urllib.request.Request(
        url,
        headers={"Accept": PYPI_SIMPLE_ACCEPT_HEADER},
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_body = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return "package_not_found", None
        return "network_error", None
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError):
        return "network_error", None

    try:
        data = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "invalid_response", None

    if not isinstance(data, dict) or not isinstance(data.get("files"), list):
        return "invalid_response", None

    return "ok", data


# --- Grouping and filtering --------------------------------------------------

def _parse_file_version(filename: str) -> Optional[Version]:
    """Extract a release Version from one PyPI Simple API file's filename,
    using PEP 440-aware parsing (never manual string splitting)."""
    if filename.endswith(".whl"):
        try:
            _, version, _, _ = parse_wheel_filename(filename)
            return version
        except (InvalidWheelFilename, InvalidVersion):
            return None

    try:
        _, version = parse_sdist_filename(filename)
        return version
    except (InvalidSdistFilename, InvalidVersion):
        return None


def _group_files_by_version(files: List[Dict[str, Any]]) -> Dict[Version, List[Dict[str, Any]]]:
    """Turn the Simple API's flat file list into one entry per release
    version. Filenames that cannot be parsed safely are skipped and logged,
    never guessed."""
    grouped: Dict[Version, List[Dict[str, Any]]] = {}

    for file_info in files:
        filename = file_info.get("filename")
        if not filename:
            continue

        version = _parse_file_version(filename)
        if version is None:
            print(
                f"[WARN] Skipping unparseable PyPI filename: {filename!r}",
                file=sys.stderr,
            )
            continue

        grouped.setdefault(version, []).append(file_info)

    return grouped


def _requires_python_specifiers(files: List[Dict[str, Any]]) -> Tuple[List[SpecifierSet], Optional[str]]:
    """Collect the valid requires-python specifiers declared across a
    version's files, and the first raw declared value (for display)."""
    specifiers: List[SpecifierSet] = []
    first_declared: Optional[str] = None

    for file_info in files:
        raw = file_info.get("requires-python")
        if not raw:
            continue

        if first_declared is None:
            first_declared = raw

        try:
            specifiers.append(SpecifierSet(raw))
        except InvalidSpecifier:
            print(
                f"[WARN] Invalid requires-python specifier {raw!r}; ignoring for compatibility checks",
                file=sys.stderr,
            )

    return specifiers, first_declared


def _evaluate_python_compatibility(
    usable_files: List[Dict[str, Any]],
    parsed_python_version: Optional[Version],
) -> Tuple[str, Optional[str]]:
    """Return (python_compatibility, requires_python) for one release,
    given only its non-yanked files.

    - "compatible": at least one usable file's requires-python is satisfied
      by the runtime Python version.
    - "incompatible": usable files declare requires-python, but none of them
      are satisfied by the runtime Python version.
    - "unknown": either the runtime Python version is not known, or no
      usable file declares requires-python at all. Never presented as
      proven compatible (docs/rag-design.md §7.3).
    """
    specifiers, requires_python = _requires_python_specifiers(usable_files)

    if parsed_python_version is None or not specifiers:
        return "unknown", requires_python

    if any(parsed_python_version in specifier for specifier in specifiers):
        return "compatible", requires_python

    return "incompatible", requires_python


def filter_candidate_versions(
    files: List[Dict[str, Any]],
    python_version: Optional[str],
    limit: int = MAX_CANDIDATE_VERSIONS,
) -> List[Dict[str, Any]]:
    """Reduce a distribution's full PyPI file list to a small, safe,
    relevant set of candidate releases, per docs/rag-design.md §7.

    Applied in order: group files into one candidate per version; drop a
    release only if *every* one of its files is yanked (evaluate the
    non-yanked files otherwise); drop pre-release/dev releases; drop
    releases whose declared requires-python explicitly rejects the given
    Python version. Releases with *no* declared requires-python are kept
    and marked "unknown", not dropped and not claimed compatible - matching
    data/prompt-tests/l5_pypi_poc_filter_fixture.json, the L5 proof of
    concept's own verified fixture. Sorted newest-first using real version
    comparison (never string comparison), capped at `limit`.
    """
    parsed_python_version: Optional[Version] = None
    if python_version is not None:
        try:
            parsed_python_version = Version(python_version)
        except InvalidVersion:
            print(
                f"[WARN] Invalid python_version {python_version!r}; treating Python compatibility as unknown",
                file=sys.stderr,
            )

    grouped = _group_files_by_version(files)

    candidates: List[Tuple[Version, Dict[str, Any]]] = []

    for version, version_files in grouped.items():
        if version.is_prerelease or version.is_devrelease:
            continue

        non_yanked_files = [f for f in version_files if not f.get("yanked")]
        if not non_yanked_files:
            # every usable file for this release is yanked
            continue

        compatibility, requires_python = _evaluate_python_compatibility(
            non_yanked_files, parsed_python_version
        )

        if compatibility == "incompatible":
            continue

        candidates.append((
            version,
            {
                "version": str(version),
                "requires_python": requires_python,
                "python_compatibility": compatibility,
                "yanked": False,
                "yanked_reason": None,
            },
        ))

    candidates.sort(key=lambda item: item[0], reverse=True)

    return [candidate for _, candidate in candidates[:limit]]


def _latest_stable_version(files: List[Dict[str, Any]]) -> Optional[str]:
    """The newest non-prerelease, non-fully-yanked version PyPI reports for
    this distribution - independent of Python-version filtering, matching
    what an unpinned install would resolve to by default."""
    grouped = _group_files_by_version(files)

    stable_versions = [
        version
        for version, version_files in grouped.items()
        if not version.is_prerelease
        and not version.is_devrelease
        and any(not f.get("yanked") for f in version_files)
    ]

    if not stable_versions:
        return None

    return str(max(stable_versions))


# --- Public entry point ------------------------------------------------------

def _empty_result(import_name: str, python_version: Optional[str]) -> Dict[str, Any]:
    return {
        "status": None,
        "import_name": import_name,
        "distribution_name": None,
        "package_found": None,
        "python_version": python_version,
        "latest_version": None,
        "candidate_versions": [],
        "source_endpoint": None,
        "retrieved_at": None,
        "error": None,
    }


def retrieve(import_name: str, python_version: Optional[str] = None) -> Dict[str, Any]:
    """Resolve `import_name` to a verified PyPI distribution, retrieve its
    release metadata, and return a bounded, safe set of candidate versions.

    `python_version` defaults to the fixed execution-runtime constant in
    config/rag_repair.yaml (currently "3.10") when not explicitly overridden
    - see docs/rag-design.md §4. Always returns the full retrieval-result
    schema (docs/rag-design.md §6), including on every failure path. Makes
    no PyPI request at all when the import name has no verified mapping.
    Never constructs or runs a pip command.
    """
    resolved_python_version = python_version if python_version is not None else DEFAULT_PYTHON_VERSION

    result = _empty_result(import_name, resolved_python_version)

    distribution_name = resolve_distribution_name(import_name)
    if distribution_name is None:
        result["status"] = "mapping_unknown"
        return result

    result["distribution_name"] = distribution_name

    normalized_name = normalize_distribution_name(distribution_name)
    endpoint = f"{PYPI_SIMPLE_BASE}/{normalized_name}/"

    fetch_status, data = fetch_pypi_project(normalized_name)

    if fetch_status == "package_not_found":
        result["status"] = "package_not_found"
        result["package_found"] = False
        result["source_endpoint"] = endpoint
        result["retrieved_at"] = utc_now()
        return result

    if fetch_status == "network_error":
        result["status"] = "network_error"
        result["source_endpoint"] = endpoint
        result["error"] = "Could not reach PyPI (timeout, DNS, or connection failure)."
        return result

    if fetch_status == "invalid_response":
        result["status"] = "invalid_response"
        result["source_endpoint"] = endpoint
        result["error"] = "PyPI response could not be parsed or was missing the expected 'files' list."
        return result

    files = data.get("files", [])

    result["package_found"] = True
    result["source_endpoint"] = endpoint
    result["retrieved_at"] = utc_now()
    result["latest_version"] = _latest_stable_version(files)

    candidates = filter_candidate_versions(files, resolved_python_version, limit=MAX_CANDIDATE_VERSIONS)
    result["candidate_versions"] = candidates

    if not candidates:
        result["status"] = "no_compatible_release"
        return result

    result["status"] = "resolved"
    return result
