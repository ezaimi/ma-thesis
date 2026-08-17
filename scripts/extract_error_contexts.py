import argparse
import csv
import json
import re
import sqlite3
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path.home() / "era/computational-reproducibility-pmc-docker/data/db/db.sqlite"
INPUT_PATH = Path("data/dependency-errors/dependency_errors.csv")
OUTPUT_DIR = Path("data/context-classification")
CONTEXTS_PATH = OUTPUT_DIR / "dependency_error_contexts.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "classification_summary.json"
VALIDATION_SAMPLE_PATH = OUTPUT_DIR / "manual_validation_sample.csv"
DOCS_PATH = Path("docs/error-classifier.md")

RAW_GITHUB_BASE = "https://raw.githubusercontent.com"
FETCH_TIMEOUT_SECONDS = 10
SURROUNDING_CELL_WINDOW = 2
MAX_IMPORT_CELLS = 5
MAX_STORED_TEXT_CHARS = 4000

# Modules whose PyPI distribution name does not match the import name.
# Used to distinguish "direct_missing_package" from "import_distribution_name_mismatch".
IMPORT_DISTRIBUTION_MISMATCHES = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "pil": "Pillow",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "skimage": "scikit-image",
    "serial": "pyserial",
    "openssl": "pyOpenSSL",
    "usb": "pyusb",
    "docx": "python-docx",
    "fitz": "PyMuPDF",
    "dotenv": "python-dotenv",
    "git": "GitPython",
}

# Local/ambiguous module names that are not installable PyPI packages.
LOCAL_AMBIGUOUS_MODULES = {"utils", "statistics", "config", "helpers", "src"}

CELL_SOURCE_PATTERN = re.compile(
    r"executing the following cell:\s*\n-+\n(.*?)\n-+\n", re.DOTALL
)
CANNOT_IMPORT_PATTERN = re.compile(
    r"cannot import name ['\"][^'\"]+['\"] from ['\"][^'\"]+['\"]", re.IGNORECASE
)

VALIDATION_SAMPLE_LIMITS = {
    "missing_package": 10,
    "wrong_version": 10,
    "system_library": 5,
    "mapping_unknown": None,   # all
    "out_of_scope": None,      # all
}
VALIDATION_SAMPLE_COLUMNS = [
    "notebook_execution_id",
    "error_type",
    "error_message",
    "failing_module",
    "original_subtype",
    "refined_subtype",
    "confidence",
    "root_cause_hint",
    "context_status",
    "manual_label",
    "notes",
]


def to_text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def normalize_error_fields(error_type, error_message, failing_module):
    return {
        "error_type_normalized": (error_type or "").strip(),
        "error_message_normalized": re.sub(r"\s+", " ", error_message or "").strip(),
        "failing_module_normalized": (failing_module or "").strip().lower(),
    }


def classify_refined(error_type, error_message, failing_module, original_subtype):
    """Refine the i1 subtype and attach a root-cause hint + confidence.

    Reuses the i1 taxonomy (missing_package, wrong_version, system_library,
    mapping_unknown, out_of_scope) but adds a root-cause hint and a
    confidence level, since i1 only decided usable vs. excluded.
    """
    message = error_message or ""
    message_lower = message.lower()
    module = (failing_module or "").strip()
    module_lower = module.lower()

    if ".so" in message or "shared object file" in message_lower:
        return "system_library", "high", "system_level_dependency"

    if module_lower in LOCAL_AMBIGUOUS_MODULES:
        refined = original_subtype if original_subtype in ("mapping_unknown", "out_of_scope") else "mapping_unknown"
        return refined, "low", "local_import_or_path_issue"

    if "cannot import name" in message_lower:
        return "wrong_version", "high", "version_or_api_incompatibility"

    if error_type == "ModuleNotFoundError" or "no module named" in message_lower:
        if not module:
            return "mapping_unknown", "low", "insufficient_context"
        if module_lower in IMPORT_DISTRIBUTION_MISMATCHES:
            return "missing_package", "high", "import_distribution_name_mismatch"
        return "missing_package", "high", "direct_missing_package"

    if "install" in message_lower or "required" in message_lower:
        if not module:
            return "mapping_unknown", "low", "insufficient_context"
        if module_lower in IMPORT_DISTRIBUTION_MISMATCHES:
            return "missing_package", "medium", "import_distribution_name_mismatch"
        return "missing_package", "medium", "direct_missing_package"

    if error_type == "ImportError":
        return "wrong_version", "medium", "transitive_dependency"

    return "out_of_scope", "low", "insufficient_context"


def legacy_hint_matches(error_type, error_message, failing_module, text):
    """Decide whether `text` (an `executions.msg` traceback) actually reports
    the SAME failure as the i1 row, not just a traceback that happens to
    mention the failing module somewhere (e.g. in the cell's source code).
    """
    error_message = error_message or ""
    error_type = error_type or ""
    module = (failing_module or "").strip()
    message_lower = error_message.lower()

    if error_message and error_message in text:
        return True

    if error_type == "ModuleNotFoundError" or "no module named" in message_lower:
        if module and (f"No module named '{module}'" in text or f'No module named "{module}"' in text):
            return True
        return False

    if "cannot import name" in message_lower:
        match = CANNOT_IMPORT_PATTERN.search(error_message)
        if match and match.group(0) in text:
            return True
        return False

    return False


def find_legacy_hint(connection, notebook_id, error_type, error_message, failing_module):
    """Tier 1: look for a matching row in the legacy `executions` table.

    `executions.msg` sometimes embeds the failing cell's source inside a
    traceback dump, but the table predates this pipeline's own error
    records and is frequently stale (see docs/error-classifier.md). A
    candidate row is only accepted if its *final reported error* matches
    the i1 row (see `legacy_hint_matches`) - merely mentioning the failing
    module somewhere in the traceback (e.g. inside the cell source) is not
    enough, since the cell may import several modules and fail on a
    different one than the i1 dataset recorded.

    Returns a tuple `(hint_or_none, was_rejected)`, where `was_rejected` is
    True when a candidate row was found (module name present) but failed
    the stricter match check.
    """
    if not failing_module:
        return None, False

    try:
        rows = connection.execute(
            "SELECT msg FROM executions WHERE notebook_id = ?", (notebook_id,)
        ).fetchall()
    except sqlite3.Error:
        return None, False

    rejected = False
    for (msg,) in rows:
        text = to_text(msg)
        if not text or failing_module not in text:
            continue

        if legacy_hint_matches(error_type, error_message, failing_module, text):
            match = CELL_SOURCE_PATTERN.search(text)
            cell_source = match.group(1).strip() if match else None
            return {
                "matched_module": failing_module,
                "cell_source": cell_source,
                "raw_traceback": text[:MAX_STORED_TEXT_CHARS],
            }, False

        rejected = True

    return None, rejected

# reads repository paths, commits and dependency-file locations from the database
def load_repository_metadata(connection):
    try:
        rows = connection.execute(
            'SELECT id, repository, "commit", requirements, setups, pipfiles, pipfile_locks FROM repositories;'
        ).fetchall()
    except sqlite3.Error:
        return {}

    return {
        row[0]: {
            "repository": row[1],
            "commit": row[2],
            "requirements": row[3],
            "setups": row[4],
            "pipfiles": row[5],
            "pipfile_locks": row[6],
        }
        for row in rows
    }

# creates a raw GitHub URL
def build_raw_url(repository, ref, path):
    return f"{RAW_GITHUB_BASE}/{repository}/{ref}/{path}"

# downloads text from that URL
def fetch_text(url):
    try:
        with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT_SECONDS) as response:
            if response.status != 200:
                return None
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, OSError):
        return None

# tries the recorded commit, then main, then master
def fetch_with_ref_fallback(repository, path, commit):
   
    if not repository or not path:
        return None, None

    refs = []
    if commit:
        refs.append(commit)
    refs.extend(["main", "master"])

    for ref in refs:
        text = fetch_text(build_raw_url(repository, ref, path))
        if text is not None:
            return text, ref

    return None, None

# converts a notebook cell’s source into one string
def cell_source_text(cell):
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return source or ""

# → What code was the notebook executing?
def extract_notebook_context(notebook_text, error_cell_index):
    """Extract failing/import/surrounding cells from a fetched .ipynb JSON.

    `error_cell_index` indexes into the *full* `cells` array (all cell
    types), matching how the pipeline's summary.py records it via
    `enumerate(notebook.cells)` - not a code-cell-only counter.
    """
    try:
        notebook = json.loads(notebook_text)
    except (json.JSONDecodeError, TypeError):
        return None

    cells = notebook.get("cells", [])
    if not cells:
        return None

    failing_cell_source = None
    if error_cell_index is not None and 0 <= error_cell_index < len(cells):
        failing_cell_source = cell_source_text(cells[error_cell_index])

    import_cells = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        text = cell_source_text(cell)
        if re.search(r"^\s*(import|from)\s+\S+", text, re.MULTILINE):
            import_cells.append({"cell_index": i, "source": text})
        if len(import_cells) >= MAX_IMPORT_CELLS:
            break

    surrounding_cells = []
    if error_cell_index is not None:
        start = max(0, error_cell_index - SURROUNDING_CELL_WINDOW)
        end = min(len(cells), error_cell_index + SURROUNDING_CELL_WINDOW + 1)
        for i in range(start, end):
            surrounding_cells.append({"cell_index": i, "source": cell_source_text(cells[i])})

    return {
        "failing_cell_source": failing_cell_source,
        "import_cells": import_cells,
        "surrounding_cells": surrounding_cells,
    }

# → What packages and versions did the repository declare?
def fetch_dependency_files(repository, commit, path_field_value):
    if not path_field_value:
        return []

    results = []
    for path in path_field_value.split(";"):
        path = path.strip()
        if not path:
            continue
        text, ref = fetch_with_ref_fallback(repository, path, commit)
        results.append({
            "path": path,
            "fetched": text is not None,
            "ref": ref,
            "content": text[:MAX_STORED_TEXT_CHARS] if text else None,
        })
    return results

# collects the information that will later be placed in the LLM prompt
def build_prompt_context(row, refined_subtype, confidence, root_cause_hint,
                          legacy_hint, remote_context, dependency_files, context_status):
    return {
        "error_type": row["error_type"],
        "error_message": row["error_message"],
        "failing_module": row["failing_module"],
        "subtype": refined_subtype,
        "confidence": confidence,
        "root_cause_hint": root_cause_hint,
        "failing_cell_source": (
            (remote_context or {}).get("failing_cell_source")
            or (legacy_hint or {}).get("cell_source")
        ),
        "import_cells": (remote_context or {}).get("import_cells", []),
        "surrounding_cells": (remote_context or {}).get("surrounding_cells", []),
        "dependency_files": dependency_files,
        "context_status": context_status,
    }


def enrich_row(row, connection, repo_meta_by_id, fetch_remote):
    failing_module = row.get("failing_module")
    original_subtype = row.get("subtype")

    refined_subtype, confidence, root_cause_hint = classify_refined(
        row.get("error_type"), row.get("error_message"), failing_module, original_subtype
    )

    legacy_hint, legacy_hint_rejected = find_legacy_hint(
        connection, int(row["notebook_id"]), row.get("error_type"), row.get("error_message"), failing_module
    )

    remote_context = None
    dependency_files = []
    remote_attempted = False
    remote_succeeded = False

    if fetch_remote:
        remote_attempted = True
        try:
            repo_meta = repo_meta_by_id.get(int(row["repository_id"]))
            if repo_meta and repo_meta.get("repository"):
                notebook_text, _ref = fetch_with_ref_fallback(
                    repo_meta["repository"], row.get("notebook_name"), repo_meta.get("commit")
                )
                if notebook_text is not None:
                    raw_index = row.get("error_cell_index")
                    error_cell_index = int(raw_index) if raw_index not in (None, "") else None
                    remote_context = extract_notebook_context(notebook_text, error_cell_index)
                    remote_succeeded = remote_context is not None

                for field in ("requirements", "setups", "pipfiles", "pipfile_locks"):
                    dependency_files.extend(
                        fetch_dependency_files(repo_meta["repository"], repo_meta.get("commit"), repo_meta.get(field))
                    )
        except Exception as exc:  # noqa: BLE001 - a remote-fetch hiccup must not discard classification/legacy-hint results already computed for this row
            print(
                f"[WARN] Remote fetch failed for notebook_execution_id={row.get('notebook_execution_id')}: {exc}",
                file=sys.stderr,
            )
            remote_context = None
            remote_succeeded = False

    if remote_succeeded:
        context_status = "remote_fetched"
    elif legacy_hint is not None:
        context_status = "legacy_execution_hint"
    elif remote_attempted:
        context_status = "source_not_found"
    else:
        context_status = "metadata_only"

    prompt_context = build_prompt_context(
        row, refined_subtype, confidence, root_cause_hint,
        legacy_hint, remote_context, dependency_files, context_status
    )

    result = {
        "notebook_execution_id": int(row["notebook_execution_id"]),
        "repository_id": int(row["repository_id"]),
        "notebook_id": int(row["notebook_id"]),
        "notebook_name": row.get("notebook_name"),
        "repository_url": row.get("repository_url"),
        "error_type": row.get("error_type"),
        "error_message": row.get("error_message"),
        "error_cell_index": row.get("error_cell_index"),
        "failing_module": failing_module,
        "original_subtype": original_subtype,
        "refined_subtype": refined_subtype,
        "scope_status": row.get("scope_status"),
        "exclusion_reason": row.get("exclusion_reason"),
        "split": row.get("split"),
        "confidence": confidence,
        "root_cause_hint": root_cause_hint,
        "context_status": context_status,
        "legacy_traceback_hint": legacy_hint,
        "legacy_hint_rejected": legacy_hint_rejected,
        "dependency_file_metadata": dependency_files,
        "prompt_context": prompt_context,
    }
    result.update(normalize_error_fields(row.get("error_type"), row.get("error_message"), failing_module))
    return result

# produces a safe basic record if normal enrichment crashes
def fallback_metadata_only(row, error):
    print(
        f"[WARN] Falling back to metadata_only for "
        f"notebook_execution_id={row.get('notebook_execution_id')}: {error}",
        file=sys.stderr,
    )

    failing_module = row.get("failing_module")
    original_subtype = row.get("subtype") or "out_of_scope"

    result = {
        "notebook_execution_id": int(row["notebook_execution_id"]) if row.get("notebook_execution_id") else None,
        "repository_id": int(row["repository_id"]) if row.get("repository_id") else None,
        "notebook_id": int(row["notebook_id"]) if row.get("notebook_id") else None,
        "notebook_name": row.get("notebook_name"),
        "repository_url": row.get("repository_url"),
        "error_type": row.get("error_type"),
        "error_message": row.get("error_message"),
        "error_cell_index": row.get("error_cell_index"),
        "failing_module": failing_module,
        "original_subtype": original_subtype,
        "refined_subtype": original_subtype,
        "scope_status": row.get("scope_status"),
        "exclusion_reason": row.get("exclusion_reason"),
        "split": row.get("split"),
        "confidence": "low",
        "root_cause_hint": "insufficient_context",
        "context_status": "metadata_only",
        "legacy_traceback_hint": None,
        "legacy_hint_rejected": False,
        "dependency_file_metadata": [],
        "prompt_context": {
            "error_type": row.get("error_type"),
            "error_message": row.get("error_message"),
            "failing_module": failing_module,
            "subtype": original_subtype,
            "confidence": "low",
            "root_cause_hint": "insufficient_context",
            "failing_cell_source": None,
            "import_cells": [],
            "surrounding_cells": [],
            "dependency_files": [],
            "context_status": "metadata_only",
        },
    }
    result.update(normalize_error_fields(row.get("error_type"), row.get("error_message"), failing_module))
    return result

# selects rows from each subtype for manual review: 5 nga nje subtype, 10 nga nje tjeter
def stratified_sample(enriched_rows):
    """Deterministic stratified sample, taking the first N rows per subtype
    in dataset order (no randomness, for reproducibility)."""
    buckets = {}
    for row in enriched_rows:
        buckets.setdefault(row["refined_subtype"], []).append(row)

    sample = []
    for subtype, items in buckets.items():
        limit = VALIDATION_SAMPLE_LIMITS.get(subtype, None)
        sample.extend(items if limit is None else items[:limit])
    return sample

# krijon manual_validation_sample.csv
def write_validation_sample(sample_rows):
    with VALIDATION_SAMPLE_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=VALIDATION_SAMPLE_COLUMNS)
        writer.writeheader()
        for row in sample_rows:
            writer.writerow({
                "notebook_execution_id": row["notebook_execution_id"],
                "error_type": row["error_type"],
                "error_message": row["error_message"],
                "failing_module": row["failing_module"],
                "original_subtype": row["original_subtype"],
                "refined_subtype": row["refined_subtype"],
                "confidence": row["confidence"],
                "root_cause_hint": row["root_cause_hint"],
                "context_status": row["context_status"],
                "manual_label": "",
                "notes": "",
            })

# calculates classification and context statistics
def build_summary(enriched_rows, sample_rows, fetch_remote):
    context_status_counts = Counter(r["context_status"] for r in enriched_rows)
    dependency_files_attempted = sum(len(r["dependency_file_metadata"]) for r in enriched_rows)
    dependency_files_fetched = sum(
        1 for r in enriched_rows for f in r["dependency_file_metadata"] if f["fetched"]
    )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fetch_remote_enabled": fetch_remote,
        "total_rows": len(enriched_rows),
        "subtype_counts_before_refinement": dict(Counter(r["original_subtype"] for r in enriched_rows)),
        "subtype_counts_after_refinement": dict(Counter(r["refined_subtype"] for r in enriched_rows)),
        "confidence_distribution": dict(Counter(r["confidence"] for r in enriched_rows)),
        "root_cause_hint_counts": dict(Counter(r["root_cause_hint"] for r in enriched_rows)),
        "context_status_counts": dict(context_status_counts),
        "rows_with_legacy_hint": sum(1 for r in enriched_rows if r["legacy_traceback_hint"] is not None),
        "rows_with_rejected_legacy_hint": sum(1 for r in enriched_rows if r.get("legacy_hint_rejected")),
        "rows_with_remote_fetched_context": context_status_counts.get("remote_fetched", 0),
        "rows_with_metadata_only_context": context_status_counts.get("metadata_only", 0),
        "dependency_file_metadata_counts": {
            "attempted": dependency_files_attempted,
            "fetched": dependency_files_fetched,
        },
        "validation_sample_size": len(sample_rows),
    }


def write_docs():
    DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS_PATH.write_text("""# Error Classifier & Context Extraction (i2)

## Purpose

This document describes the ErrorClassifier and context-extraction component built on top of
the i1 dependency-error dataset. It refines each row's subtype, attaches a root-cause hint and
confidence level, and attempts to recover source context (failing cell, nearby imports,
dependency files) for later use by the LLM explanation/repair prompts.

## Input Files

- `data/dependency-errors/dependency_errors.csv` (i1 output)
- `~/era/computational-reproducibility-pmc-docker/data/db/db.sqlite`, tables `executions` and `repositories`

## Output Files

- `data/context-classification/dependency_error_contexts.jsonl` - one enriched record per i1 row
- `data/context-classification/classification_summary.json` - counts and distributions
- `data/context-classification/manual_validation_sample.csv` - stratified sample for manual agreement checks

Every enriched record carries three fields through unchanged from the i1 CSV: `scope_status`
(`usable` / `excluded`), `exclusion_reason`, and `split` (`dev` / `evaluation` / `excluded`). These
are i1's repair-eligibility classification, not part of i2's own subtype refinement - i2 still
enriches and writes a record for every i1 row regardless of `scope_status`. The fields exist so
that a later `RAGRepairAgent` (i4) can decide whether to attempt repair without re-deriving
eligibility, while `LLMExplainer` (i3) continues to explain every row. See
`docs/architecture-note.md` §7.1 for the full explanation-vs-repair scope split.

## Classifier Rules

Subtypes reuse the i1 taxonomy: `missing_package`, `wrong_version`, `system_library`,
`mapping_unknown`, `out_of_scope`. Rules are applied in this order:

1. `.so` / "shared object file" in the message -> `system_library`
2. Failing module is a known ambiguous local name (`utils`, `statistics`, `config`, `helpers`, `src`) -> `mapping_unknown` (or kept as `out_of_scope` if i1 already excluded it for that reason)
3. "cannot import name" in the message -> `wrong_version`
4. `ModuleNotFoundError` / "no module named" -> `missing_package`
5. "install" / "required" phrasing (e.g. optional-dependency messages) -> `missing_package`
6. Any other `ImportError` -> `wrong_version` (treated as a transitive-dependency signal)
7. Anything else, or a missing failing module -> `out_of_scope` / `mapping_unknown` with `insufficient_context`

## Root-Cause Hints

- `direct_missing_package` - the failing module is genuinely absent and its import name matches its PyPI distribution name
- `import_distribution_name_mismatch` - the import name differs from the PyPI distribution name (e.g. `sklearn` -> `scikit-learn`, see `IMPORT_DISTRIBUTION_MISMATCHES` in the script)
- `version_or_api_incompatibility` - "cannot import name" style failures
- `transitive_dependency` - generic `ImportError` not matching the more specific patterns above
- `system_level_dependency` - missing shared object / system library
- `local_import_or_path_issue` - ambiguous local module names
- `insufficient_context` - not enough information in the error message to decide

## Confidence Logic

- `high` - direct pattern match on error type/message (missing-module, cannot-import-name, `.so` failures)
- `medium` - inferred from softer phrasing ("install", "required") or a generic `ImportError`
- `low` - ambiguous local modules or an empty/unusable failing module

## Context Extraction Tiers

1. **`legacy_execution_hint`** - look up the notebook in the legacy `executions` table (by `notebook_id`). A candidate row must first mention the `failing_module` string somewhere, then pass a stricter check that the row's *final reported error* actually matches the i1 row, not just its cell source: either (a) the exact i1 `error_message` appears in `executions.msg`, or (b) for `ModuleNotFoundError`, the exact pattern `No module named '<failing_module>'` (or double-quoted) appears, or (c) for "cannot import name" errors, the same `cannot import name '...' from '...'` phrase appears. A row that only mentions the module inside the cell's import statements - while the traceback's actual final error is for a different module - is rejected and counted in `rows_with_rejected_legacy_hint`. The embedded cell source (when accepted) is extracted from the "executing the following cell:" traceback block. Always low-confidence, never authoritative.
2. **`remote_fetched`** - only with `--fetch-remote`. Fetches the notebook JSON and dependency files from `raw.githubusercontent.com`, using the `repositories` table's `repository` (`owner/repo`) field, trying the stored `commit` first, then `main`, then `master`. Extracts the failing cell (by `error_cell_index`), nearby import cells, and a small window of surrounding cells.
3. **`metadata_only`** / **`source_not_found`** - `metadata_only` when no fetch was attempted at all; `source_not_found` when `--fetch-remote` was used but every fetch failed. Both still carry the full metadata already present in the i1 CSV (IDs, error type/message/cell index, failing module, subtype).

## Why `executions.msg` Is Only a Hint

`executions` is a legacy table with no timestamp column, inherited from an earlier/different
notebook-execution pass rather than this Docker pipeline's own re-execution. Cross-checking it
against the i1 dataset shows 162 of 214 notebooks (76%) have a matching `notebook_id`, but only
about 40% of those actually contain the expected `failing_module` string anywhere in the traceback
text - the rest show a different `reason` (`NameError`, `Exception`, `<Skipping notebook>`) or
reference a different module entirely.

Even a substring match on the module name is not enough: the module can appear in the *cell
source* (an import statement) while the traceback's actual final error is for a different module
in the same cell - e.g. a cell importing `pandas` before `sklearn` may fail on `pandas` while the
current i1 row records a `sklearn` failure for the same notebook. `find_legacy_hint` therefore
only accepts a row once its final reported error matches the i1 row (see the tier description
above); rejected candidates are counted separately in `rows_with_rejected_legacy_hint` so the gap
between "module mentioned" and "module actually failed" stays visible.

## Why Metadata-Only Context Is Sometimes Necessary

No actual notebook source is stored locally: the `notebooks`/`repositories` tables hold only
metadata (cell counts, hashes, semicolon-separated path lists), never file contents, and the
repositories cloned during the original Docker pipeline run were never persisted to this machine
(`data/repositories/` does not exist). Falling back to metadata-only context lets the pipeline
keep every row instead of dropping ones without recoverable source.

## Remote-Fetch Limitations

- The pipeline does not pin to the recorded `commit` when cloning; this script tries the commit
  first but falls back to `main`/`master`, so a fetched notebook may not exactly match the code
  that originally failed.
- No GitHub authentication is used, so fetching is subject to `raw.githubusercontent.com`'s
  anonymous rate limits if run over the full dataset repeatedly.
- `error_cell_index` indexes into the full `cells` array of the notebook JSON (all cell types,
  per the pipeline's own `enumerate(notebook.cells)` logic in `summary.py`), not a code-cell-only
  counter - this script indexes the same way, but a notebook edited since the original run can
  still shift cells and misalign the index.
- Off by default; only runs with `--fetch-remote`, and every fetch failure falls through silently
  to the next tier rather than raising.

## Known Edge Cases

- Repository renamed, deleted, or made private since the original execution
- Notebook edited after `error_cell_index` was recorded, shifting cell positions
- Empty or missing `failing_module` (falls back to `mapping_unknown` / `insufficient_context`)
- A `notebook_id` match in `executions` with no corresponding `failing_module` mention (ignored, tier falls through)
- Any unexpected per-row error (bad row data, DB error) falls back to a bare `metadata_only` record instead of failing the whole batch

The extraction script is `scripts/extract_error_contexts.py`.
""", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Refine i1 dependency-error subtypes and extract source context (i2)."
    )
    parser.add_argument(
        "--fetch-remote", action="store_true",
        help="Attempt best-effort GitHub raw fetches for notebook source and dependency files."
    )
    parser.add_argument(
        "--db-path", type=Path, default=DB_PATH,
        help="Path to the Docker pipeline's db.sqlite (default: ~/era/computational-reproducibility-pmc-docker/data/db/db.sqlite)."
    )
    parser.add_argument(
        "--input", type=Path, default=INPUT_PATH,
        help="Path to the i1 dependency_errors.csv."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(args.db_path)
    repo_meta_by_id = load_repository_metadata(connection)

    with args.input.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    enriched_rows = []
    for row in rows:
        try:
            enriched_rows.append(enrich_row(row, connection, repo_meta_by_id, args.fetch_remote))
        except Exception as exc:  # noqa: BLE001 - one bad row must not stop the batch
            enriched_rows.append(fallback_metadata_only(row, exc))

    connection.close()

    with CONTEXTS_PATH.open("w", encoding="utf-8") as file:
        for enriched in enriched_rows:
            file.write(json.dumps(enriched) + "\n")

    sample_rows = stratified_sample(enriched_rows)
    write_validation_sample(sample_rows)

    summary = build_summary(enriched_rows, sample_rows, args.fetch_remote)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    write_docs()

    print(f"Loaded rows: {len(rows)}")
    print(f"Remote fetch enabled: {args.fetch_remote}")
    print(f"Wrote: {CONTEXTS_PATH}")
    print(f"Wrote: {SUMMARY_PATH}")
    print(f"Wrote: {VALIDATION_SAMPLE_PATH}")
    print(f"Wrote: {DOCS_PATH}")


if __name__ == "__main__":
    main()
