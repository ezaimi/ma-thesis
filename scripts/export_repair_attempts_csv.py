#!/usr/bin/env python3

"""Export the repair_attempts SQLite table (scripts/result_logger.py) to a
CSV file shaped for the FAIR Jupyter KG's RML mapping convention
(mapping/rml_mapping/repair_attempts.rml.ttl) - every existing mapping in
that KG reads from a plain CSV via rml:referenceFormulation ql:CSV, never
from a live database connection.

repair_attempts itself only stores notebook_execution_id (the Docker
pipeline's own execution-attempt id - see docs/architecture-note.md
section 6.2). The KG identifies a notebook by notebook_id instead (its
notebooks.rml.ttl subject template is notebook_{id}, and the Part 2
investigation confirmed notebook_id/repository_id are the exact same
primary keys the Docker pipeline and the FAIR Jupyter KG both use for the
same notebook - see docs/architecture-note.md section 6.3/9 and
docs/result-logger.md section 8). This export recovers notebook_id per row the same
way FixApplicator already recovers repository/notebook metadata: by
re-joining notebook_execution_id against the i2 dataset
(scripts.fix_applicator.load_i2_index). notebook_id and repository_id were
confirmed (across all 214 real dependency-error rows, reproducibly -
see scripts/validate_kg_notebook_alignment.py) to be the exact same
primary keys the Docker pipeline and the FAIR Jupyter KG both use for the
same notebook, so no crosswalk table is needed. A notebook_execution_id missing
from the i2 dataset is a hard error, not a silently-null column - it
should never happen in practice, since log_repair_attempts() already
requires every notebook_execution_id it logs to resolve against that same
i2 dataset (see scripts/result_logger.py).
"""

import argparse
import csv
import sqlite3
from pathlib import Path
from typing import List

from fix_applicator import load_i2_index
from result_logger import REPAIR_ATTEMPT_COLUMNS


CSV_COLUMNS: List[str] = ["id", "notebook_id"] + REPAIR_ATTEMPT_COLUMNS


def export_repair_attempts_csv(conn: sqlite3.Connection, i2_path: str, output_path: str) -> int:
    """Write one CSV row per repair_attempts row, with notebook_id resolved
    via the i2 dataset. Returns the number of rows written."""
    i2_index = load_i2_index(i2_path)

    select_columns = ["id"] + REPAIR_ATTEMPT_COLUMNS
    rows = conn.execute(
        "SELECT {} FROM repair_attempts ORDER BY id".format(", ".join(select_columns))
    ).fetchall()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            record = dict(zip(select_columns, row))
            notebook_execution_id = record["notebook_execution_id"]
            i2_row = i2_index.get(int(notebook_execution_id))
            if i2_row is None:
                raise ValueError(
                    f"repair_attempts.id={record['id']}: notebook_execution_id="
                    f"{notebook_execution_id} not found in i2 dataset {i2_path} - "
                    "cannot resolve notebook_id for the KG link"
                )
            record["notebook_id"] = i2_row.get("notebook_id")
            writer.writerow({column: record[column] for column in CSV_COLUMNS})
            written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export repair_attempts to a KG-ready CSV (notebook_id resolved via the i2 dataset)."
    )
    parser.add_argument("--db-path", required=True)
    parser.add_argument(
        "--i2", default="data/context-classification/dependency_error_contexts.jsonl"
    )
    parser.add_argument("--output", default="data/repair-attempts/repair_attempts.csv")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    try:
        written = export_repair_attempts_csv(conn, args.i2, args.output)
        print(f"exported {written} row(s) to {args.output}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
