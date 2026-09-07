#!/usr/bin/env python3

"""Reproduce the i6 "no crosswalk needed" claim (docs/result-logger.md
section 8): for every row of the i2 classification dataset, confirm that
its Docker-pipeline `repository_id`/`notebook_id` resolve to the same
repository and notebook identity in the external FAIR Jupyter KG's own
`repositories.csv`/`notebooks.csv`.

The external FAIR Jupyter KG checkout is not part of this repository (its
CSVs are large, and it is a separate upstream project - see
docs/result-logger.md section 8). This script takes its two source files
as CLI arguments rather than assuming a fixed location, so it works
against whatever checkout path the caller has locally.

Identity check per i2 row, mirroring the two joins
`mapping/rml_mapping/notebooks.rml.ttl` (in that external checkout) itself
performs:

- repository_id -> repositories.csv row with matching `id`; its
  `domain`/`repository` columns must reconstruct the same GitHub URL i2
  recorded as `repository_url`.
- notebook_id -> notebooks.csv row with matching `id`; its `name` column
  must equal i2's `notebook_name`, AND that notebook row's own
  `repository_id` column must equal the same repository_id (this is what
  actually establishes "the same notebook", not just an id that happens to
  exist) - notebooks.rml.ttl's own join (`map:jc_000`) relies on exactly
  this notebooks.repository_id -> repositories.id relationship.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fix_applicator import load_i2_index


class DuplicateSourceIdError(Exception):
    """Raised when an external KG CSV has two rows sharing the same `id` -
    malformed reference data that must never be silently collapsed by
    picking whichever row happened to load last."""


def load_csv_by_id(path: str, id_column: str = "id") -> Dict[int, Dict[str, str]]:
    """Load an external FAIR Jupyter KG CSV keyed by its own `id` column."""
    index: Dict[int, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_id = row.get(id_column)
            if raw_id is None or raw_id == "":
                continue
            row_id = int(raw_id)
            if row_id in index:
                raise DuplicateSourceIdError(
                    f"{path}: duplicate {id_column}={row_id} - refusing to silently "
                    "pick one row over the other"
                )
            index[row_id] = row
    return index


def normalize_repository_url(domain: str, repository: str) -> str:
    """Reconstruct the GitHub URL i2's own `repository_url` records, from
    repositories.csv's separate `domain`/`repository` columns."""
    return f"https://{(domain or '').strip()}/{(repository or '').strip()}".rstrip("/")


def check_record(
    i2_row: Dict[str, Any],
    repositories_index: Dict[int, Dict[str, str]],
    notebooks_index: Dict[int, Dict[str, str]],
) -> Dict[str, Any]:
    """Check one i2 row's repository_id/notebook_id against the external
    KG's own source tables. Returns a per-record result dict; never raises
    on a data mismatch - only DuplicateSourceIdError (raised earlier, while
    loading) is treated as fatal."""
    repository_id = i2_row.get("repository_id")
    notebook_id = i2_row.get("notebook_id")

    result: Dict[str, Any] = {
        "notebook_execution_id": i2_row.get("notebook_execution_id"),
        "repository_id": repository_id,
        "notebook_id": notebook_id,
        "repository_status": "missing",
        "notebook_status": "missing",
        "detail": None,
    }

    repo_row = repositories_index.get(int(repository_id)) if repository_id is not None else None
    if repo_row is None:
        result["repository_status"] = "missing"
    else:
        expected_url = normalize_repository_url(repo_row.get("domain", ""), repo_row.get("repository", ""))
        actual_url = (i2_row.get("repository_url") or "").rstrip("/")
        if expected_url.lower() == actual_url.lower():
            result["repository_status"] = "match"
        else:
            result["repository_status"] = "mismatch"
            result["detail"] = f"repository: KG resolves to {expected_url!r}, i2 has {actual_url!r}"

    nb_row = notebooks_index.get(int(notebook_id)) if notebook_id is not None else None
    if nb_row is None:
        result["notebook_status"] = "missing"
    else:
        expected_name = (nb_row.get("name") or "").strip()
        actual_name = (i2_row.get("notebook_name") or "").strip()
        nb_repository_id = nb_row.get("repository_id")
        same_repository = (
            repository_id is not None
            and nb_repository_id is not None
            and int(nb_repository_id) == int(repository_id)
        )
        if expected_name == actual_name and same_repository:
            result["notebook_status"] = "match"
        else:
            result["notebook_status"] = "mismatch"
            reasons = []
            if expected_name != actual_name:
                reasons.append(f"name {expected_name!r} != i2's {actual_name!r}")
            if not same_repository:
                reasons.append(
                    f"notebook's own repository_id {nb_repository_id!r} != i2's {repository_id!r}"
                )
            note = "notebook: " + "; ".join(reasons)
            result["detail"] = f"{result['detail']}; {note}" if result["detail"] else note

    return result


def validate_alignment(
    i2_rows: List[Dict[str, Any]],
    repositories_index: Dict[int, Dict[str, str]],
    notebooks_index: Dict[int, Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Check every i2 row and return the full list of per-record results,
    in the given row order."""
    return [check_record(row, repositories_index, notebooks_index) for row in i2_rows]


def summarize(results: List[Dict[str, Any]]) -> Dict[str, int]:
    total = len(results)
    repository_matches = sum(1 for r in results if r["repository_status"] == "match")
    notebook_matches = sum(1 for r in results if r["notebook_status"] == "match")
    missing = sum(
        1 for r in results if r["repository_status"] == "missing" or r["notebook_status"] == "missing"
    )
    mismatches = sum(
        1
        for r in results
        if r["repository_status"] != "missing"
        and r["notebook_status"] != "missing"
        and (r["repository_status"] == "mismatch" or r["notebook_status"] == "mismatch")
    )
    return {
        "total": total,
        "repository_matches": repository_matches,
        "notebook_matches": notebook_matches,
        "missing": missing,
        "mismatches": mismatches,
    }


def print_report(results: List[Dict[str, Any]], summary: Dict[str, int]) -> None:
    for r in results:
        if r["repository_status"] != "match" or r["notebook_status"] != "match":
            print(
                f"FAIL notebook_execution_id={r['notebook_execution_id']} "
                f"repository_id={r['repository_id']} (repository: {r['repository_status']}) "
                f"notebook_id={r['notebook_id']} (notebook: {r['notebook_status']})"
                + (f" - {r['detail']}" if r["detail"] else "")
            )

    print(f"Records checked: {summary['total']}")
    print(f"Repository matches: {summary['repository_matches']}/{summary['total']}")
    print(f"Notebook matches: {summary['notebook_matches']}/{summary['total']}")
    print(f"Missing: {summary['missing']}")
    print(f"Mismatches: {summary['mismatches']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that this repo's i2 dataset repository_id/notebook_id values "
            "resolve to the same repository/notebook identity in the external FAIR "
            "Jupyter KG's repositories.csv/notebooks.csv."
        )
    )
    parser.add_argument(
        "--i2", default="data/context-classification/dependency_error_contexts.jsonl"
    )
    parser.add_argument(
        "--repositories-csv",
        required=True,
        help="Path to the external FAIR Jupyter KG checkout's data/repositories.csv",
    )
    parser.add_argument(
        "--notebooks-csv",
        required=True,
        help="Path to the external FAIR Jupyter KG checkout's data/notebooks.csv",
    )
    args = parser.parse_args()

    i2_index = load_i2_index(args.i2)
    i2_rows = [i2_index[key] for key in sorted(i2_index.keys())]

    repositories_index = load_csv_by_id(args.repositories_csv)
    notebooks_index = load_csv_by_id(args.notebooks_csv)

    results = validate_alignment(i2_rows, repositories_index, notebooks_index)
    summary = summarize(results)
    print_report(results, summary)

    if summary["missing"] > 0 or summary["mismatches"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
