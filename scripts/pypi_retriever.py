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
USER_AGENT = "ma-thesis-rag-repair-agent/i4 (Python dependency-repair research prototype)"
DEFAULT_FETCH_TIMEOUT_SECONDS = 10.0
MAX_CANDIDATE_VERSIONS = 5

# PEP 503-normalized project names: lowercase alphanumerics joined by single
# hyphens. normalize_distribution_name() always produces this shape; this
# regex is a defensive check before any name is inserted into a URL.
_NORMALIZED_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


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
    never looked up per notebook. `path` lets tests point at a temporary
    file to exercise missing/malformed-configuration behavior."""
    config_path = Path(path) if path else DEFAULT_RAG_REPAIR_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def validate_python_version_config(value: Any) -> Tuple[Optional[str], Optional[str]]:
    """Validate one runtime.python_version value from config/rag_repair.yaml.

    Returns (version_string, None) if valid, or (None, error_message) if the
    value is missing, not a string, or not a valid PEP 440 version. Never
    raises - callers use the error message to build a structured
    "configuration_error" retrieval result instead of crashing or silently
    falling back to some other version source (e.g. notebook metadata).
    """
    if not value or not isinstance(value, str):
        return None, "config/rag_repair.yaml: runtime.python_version is missing or not a string"

    try:
        Version(value)
    except InvalidVersion:
        return None, f"config/rag_repair.yaml: runtime.python_version {value!r} is not a valid PEP 440 version"

    return value, None


_RAG_REPAIR_CONFIG = load_rag_repair_config()
DEFAULT_PYTHON_VERSION, DEFAULT_PYTHON_VERSION_ERROR = validate_python_version_config(
    _RAG_REPAIR_CONFIG.get("runtime", {}).get("python_version")
)


def resolve_distribution_name(import_name: str) -> Optional[str]:
    return _PACKAGE_MAPPING.get(import_name)


def normalize_distribution_name(distribution_name: str) -> str:
    return re.sub(r"[-_.]+", "-", distribution_name).lower()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- PyPI Simple API client -------------------------------------------------

_pypi_response_cache: Dict[str, Tuple[str, Optional[Dict[str, Any]]]] = {}


def clear_pypi_cache() -> None:
    """Reset the in-memory fetch cache. Exposed so tests don't leak cached
    responses between cases, and so a long batch run can be reset between
    independent phases if ever needed."""
    _pypi_response_cache.clear()


def fetch_pypi_project(
    distribution_name: str,
    timeout: float = DEFAULT_FETCH_TIMEOUT_SECONDS,
    urlopen: Optional[Any] = None,
    use_cache: bool = True,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Fetch one distribution's file listing from the official PyPI JSON
    Simple API (PEP 691), https://pypi.org/simple/{name}/ - never any other
    host, and never a URL supplied by a caller, an LLM, or an input record.

    `distribution_name` must already be PEP 503-normalized; it is validated
    again here before being inserted into the URL. `urlopen` is injectable
    for tests (default is the real urllib.request.urlopen, resolved at call
    time so monkeypatching urllib.request.urlopen also still works).

    Returns (status, data): status is "ok" (data is the parsed JSON dict),
    "package_not_found", "network_error", or "invalid_response" (data is
    None in the latter three cases). Distinguishes HTTP, connection,
    timeout, and parsing failures rather than collapsing them into one
    generic error, per day-1-rag-repair-agent-plan.md Step 4.

    Within-process results are cached by normalized distribution name
    (`use_cache=True`, the default) so a batch run never issues two
    requests for the same project - see docs/rag-design.md §9.3.
    """
    if use_cache and distribution_name in _pypi_response_cache:
        return _pypi_response_cache[distribution_name]

    if not _NORMALIZED_NAME_RE.match(distribution_name):
        result = ("invalid_response", None)
        if use_cache:
            _pypi_response_cache[distribution_name] = result
        return result

    url = f"{PYPI_SIMPLE_BASE}/{distribution_name}/"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": PYPI_SIMPLE_ACCEPT_HEADER,
            "User-Agent": USER_AGENT,
        },
    )

    opener = urlopen if urlopen is not None else urllib.request.urlopen

    try:
        with opener(request, timeout=timeout) as response:
            raw_body = response.read()
    except urllib.error.HTTPError as exc:
        result = ("package_not_found", None) if exc.code == 404 else ("network_error", None)
        if use_cache:
            _pypi_response_cache[distribution_name] = result
        return result
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError):
        result = ("network_error", None)
        if use_cache:
            _pypi_response_cache[distribution_name] = result
        return result

    try:
        data = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        result = ("invalid_response", None)
        if use_cache:
            _pypi_response_cache[distribution_name] = result
        return result

    if not isinstance(data, dict) or not isinstance(data.get("files"), list):
        result = ("invalid_response", None)
    else:
        result = ("ok", data)

    if use_cache:
        _pypi_response_cache[distribution_name] = result
    return result


# --- Parsing and grouping ----------------------------------------------------

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


def _group_files_by_version(
    files: List[Dict[str, Any]],
    warnings: Optional[List[str]] = None,
) -> Dict[Version, List[Dict[str, Any]]]:
    """Turn the Simple API's flat file list into one entry per release
    version. Filenames that cannot be parsed safely, or non-dict file
    entries, are skipped and recorded - never guessed. Only the metadata
    needed for later filtering/evidence logging (filename, yanked,
    requires-python) survives; nothing here infers API-level compatibility
    from a filename or any PyPI metadata."""
    grouped: Dict[Version, List[Dict[str, Any]]] = {}

    for file_info in files:
        if not isinstance(file_info, dict):
            message = f"Skipping non-object PyPI file entry: {file_info!r}"
            print(f"[WARN] {message}", file=sys.stderr)
            if warnings is not None:
                warnings.append(message)
            continue

        filename = file_info.get("filename")
        if not filename:
            continue

        version = _parse_file_version(filename)
        if version is None:
            message = f"Skipping unparseable PyPI filename: {filename!r}"
            print(f"[WARN] {message}", file=sys.stderr)
            if warnings is not None:
                warnings.append(message)
            continue

        grouped.setdefault(version, []).append(file_info)

    return grouped


def _requires_python_specifiers(
    files: List[Dict[str, Any]],
    warnings: Optional[List[str]] = None,
) -> Tuple[List[SpecifierSet], Optional[str]]:
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
            message = f"Invalid requires-python specifier {raw!r}; ignoring for compatibility checks"
            print(f"[WARN] {message}", file=sys.stderr)
            if warnings is not None:
                warnings.append(message)

    return specifiers, first_declared


def _evaluate_python_compatibility(
    usable_files: List[Dict[str, Any]],
    parsed_python_version: Optional[Version],
    warnings: Optional[List[str]] = None,
) -> Tuple[str, Optional[str]]:
    """Return (python_compatibility, requires_python) for one release,
    given only its non-yanked files.

    - "compatible": at least one usable file's requires-python is satisfied
      by the runtime Python version. This is PyPI-metadata compatibility
      only - it says the release *can be installed*, not that any specific
      imported symbol works, which is a distinct, stronger claim only
      wrong_version's compatibility-evidence intersection can make.
    - "incompatible": usable files declare requires-python, but none of them
      are satisfied by the runtime Python version.
    - "unknown": either the runtime Python version is not known, or no
      usable file declares requires-python at all. Never presented as
      proven compatible (docs/rag-design.md §7.3).
    """
    specifiers, requires_python = _requires_python_specifiers(usable_files, warnings)

    if parsed_python_version is None or not specifiers:
        return "unknown", requires_python

    if any(parsed_python_version in specifier for specifier in specifiers):
        return "compatible", requires_python

    return "incompatible", requires_python


def _parse_python_version_arg(
    python_version: Optional[str],
    warnings: Optional[List[str]] = None,
) -> Optional[Version]:
    if python_version is None:
        return None
    try:
        return Version(python_version)
    except InvalidVersion:
        message = f"Invalid python_version {python_version!r}; treating Python compatibility as unknown"
        print(f"[WARN] {message}", file=sys.stderr)
        if warnings is not None:
            warnings.append(message)
        return None


def _filter_grouped_candidates(
    grouped: Dict[Version, List[Dict[str, Any]]],
    parsed_python_version: Optional[Version],
    limit: Optional[int],
    warnings: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Reduce already-grouped releases to PyPI-metadata-safe candidates, per
    docs/rag-design.md §7: drop a release only if *every* one of its files
    is yanked; drop pre-release/dev releases; drop releases whose declared
    requires-python explicitly rejects the given Python version. Releases
    with *no* declared requires-python are kept and marked "unknown", not
    dropped and not claimed compatible - matching
    data/prompt-tests/l5_pypi_poc_filter_fixture.json, the L5 proof of
    concept's own verified fixture (day-1-rag-repair-agent-plan.md's
    stricter prose is superseded here; see docs/rag-design.md §2.2/§7).

    Sorted newest-first using real version comparison. `limit=None` returns
    every safe candidate, uncapped - required for the wrong_version path,
    which must intersect against the *full* safe set before any five-item
    cap is applied (capping first could silently discard an older,
    API-compatible release in favour of newer ones that turn out
    API-incompatible). `limit=<int>` caps immediately, for direct/standalone
    callers that have no further intersection step.
    """
    candidates: List[Tuple[Version, Dict[str, Any]]] = []

    for version, version_files in grouped.items():
        if version.is_prerelease or version.is_devrelease:
            continue

        non_yanked_files = [f for f in version_files if not f.get("yanked")]
        if not non_yanked_files:
            # every usable file for this release is yanked
            continue

        compatibility, requires_python = _evaluate_python_compatibility(
            non_yanked_files, parsed_python_version, warnings
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
    ordered = [candidate for _, candidate in candidates]

    if limit is None:
        return ordered
    return ordered[:limit]


def filter_candidate_versions(
    files: List[Dict[str, Any]],
    python_version: Optional[str],
    limit: Optional[int] = MAX_CANDIDATE_VERSIONS,
) -> List[Dict[str, Any]]:
    """Public, standalone entry point: group raw PyPI files and reduce them
    to PyPI-metadata-safe candidates in one call. See
    _filter_grouped_candidates() for the filtering rules. retrieve() itself
    groups once internally (see below) to avoid grouping/warning twice for
    the same fetch."""
    parsed_python_version = _parse_python_version_arg(python_version)
    grouped = _group_files_by_version(files)
    return _filter_grouped_candidates(grouped, parsed_python_version, limit)


def _latest_stable_version_from_grouped(grouped: Dict[Version, List[Dict[str, Any]]]) -> Optional[str]:
    """The newest non-prerelease, non-fully-yanked version PyPI reports for
    this distribution - independent of Python-version filtering, matching
    what an unpinned install would resolve to by default."""
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

def _empty_result(
    import_name: str,
    subtype: str,
    python_version: Optional[str],
    module_path: Optional[str],
    symbol: Optional[str],
) -> Dict[str, Any]:
    return {
        "status": None,
        "import_name": import_name,
        "subtype": subtype,
        "module_path": module_path,
        "symbol": symbol,
        "distribution_name": None,
        "package_found": None,
        "python_version": python_version,
        "latest_version": None,
        "candidate_versions": [],
        "compatibility_evidence": None,
        "source_endpoint": None,
        "retrieved_at": None,
        "warnings": [],
        "error": None,
    }


def retrieve(
    import_name: str,
    python_version: Optional[str] = None,
    subtype: str = "missing_package",
    module_path: Optional[str] = None,
    symbol: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve `import_name` to a verified PyPI distribution, retrieve its
    release metadata, and return a bounded, safe set of candidate versions.

    Deviation from day-1-rag-repair-agent-plan.md's original two-parameter
    sketch (`retrieve(import_name, python_version=None)`): `subtype`,
    `module_path`, and `symbol` are added so this single entry point can
    also perform the wrong_version compatibility-evidence intersection
    docs/rag-design.md §2.2 requires - a second, parallel function was not
    introduced. Calls that omit them behave exactly as the original
    two-parameter signature (subtype defaults to "missing_package").

    `python_version` defaults to the fixed execution-runtime constant in
    config/rag_repair.yaml (currently "3.10") when not explicitly
    overridden. If that configuration is missing or invalid and no explicit
    override is given, returns status "configuration_error" without making
    any PyPI request - never silently falls back to notebook metadata or a
    second hardcoded version.

    For `subtype="wrong_version"`, both `module_path` and `symbol` are
    required. The result's `candidate_versions` is the intersection of the
    PyPI-metadata-safe set and the officially-sourced compatibility
    evidence in config/api_compatibility_evidence.yaml - never PyPI
    metadata alone. If evidence is missing, invalid, or the intersection is
    empty, `candidate_versions` is `[]` and `status` is
    "no_compatible_release"; the LLM repair stage proposes from whatever
    survives, this function never picks or claims a single final version.

    Always returns the full schema (see module docstring / docs/rag-design.md
    §6), including on every failure path. Makes no PyPI request at all when
    the import name has no verified mapping. Never constructs, returns, or
    runs an installation command.
    """
    result = _empty_result(import_name, subtype, python_version, module_path, symbol)

    if python_version is not None:
        resolved_python_version = python_version
    elif DEFAULT_PYTHON_VERSION is not None:
        resolved_python_version = DEFAULT_PYTHON_VERSION
    else:
        result["status"] = "configuration_error"
        result["error"] = DEFAULT_PYTHON_VERSION_ERROR
        return result

    result["python_version"] = resolved_python_version

    if subtype == "wrong_version" and (not module_path or not symbol):
        # Pure caller-input validation: this can be decided before any
        # network activity, exactly like mapping_unknown below.
        result["status"] = "no_compatible_release"
        result["error"] = "wrong_version requires both module_path and symbol; none were supplied"
        return result

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
    warnings: List[str] = []

    result["package_found"] = True
    result["source_endpoint"] = endpoint
    result["retrieved_at"] = utc_now()

    grouped = _group_files_by_version(files, warnings)
    result["latest_version"] = _latest_stable_version_from_grouped(grouped)

    parsed_python_version = _parse_python_version_arg(resolved_python_version, warnings)
    general_candidates = _filter_grouped_candidates(grouped, parsed_python_version, limit=None, warnings=warnings)

    if subtype == "wrong_version":
        # Lazy import: compatibility_evidence imports normalize_distribution_name
        # from this module, so a top-level import here would be circular.
        from compatibility_evidence import (
            filter_versions_by_compatibility_evidence,
            lookup_compatibility_evidence,
        )

        evidence = lookup_compatibility_evidence(distribution_name, module_path, symbol)
        result["compatibility_evidence"] = {
            "status": evidence["status"],
            "compatible_specifier": evidence.get("compatible_specifier"),
            "evidence": evidence.get("evidence"),
        }

        if evidence["status"] != "resolved":
            result["error"] = (
                f"No usable API-compatibility evidence for {module_path}.{symbol} "
                f"(compatibility_evidence status: {evidence['status']})"
            )
            result["warnings"] = warnings
            result["status"] = "no_compatible_release"
            return result

        # Intersect against the FULL (uncapped) PyPI-safe set, then cap -
        # capping before this point could discard an older, still-compatible
        # release in favour of newer ones that later prove API-incompatible.
        candidates = filter_versions_by_compatibility_evidence(general_candidates, evidence)[:MAX_CANDIDATE_VERSIONS]
    else:
        candidates = general_candidates[:MAX_CANDIDATE_VERSIONS]

    result["candidate_versions"] = candidates
    result["warnings"] = warnings

    if not candidates:
        result["status"] = "no_compatible_release"
        return result

    result["status"] = "resolved"
    return result
