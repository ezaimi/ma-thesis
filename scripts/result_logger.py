#!/usr/bin/env python3

"""ResultLogger (i6, Part 1): join the i2/i3/i4/i5 per-record JSONL outputs
into one repair_attempts row per repair attempt and persist it to SQLite.

Grain: one row per i4 record. Each line of RAGRepairAgent's output file is
one repair attempt (including abstained/failed ones - ErrorClassifier and
RAGRepairAgent run over every DEPENDENCY_ERROR row regardless of repair
eligibility, they just abstain for out-of-scope rows), so an i4 file that
covers the full dataset already has one entry per notebook per
experimental config. This matches the table's own contract in
docs/architecture-note.md section 6.2: "one row per attempt ... re-running
the same targets under a different model or prompt produces further rows."

i4 and i5 are joined by *position*, never by notebook_execution_id:
scripts/fix_applicator.py stamps result["index"] with the 0-based line
number of the i4 input file it processed. A notebook_execution_id-keyed
join would be ambiguous - the real data/repair-proposals/i4_live_pilot.jsonl
fixture already contains two separate attempts for notebook_execution_id=8
(a timed-out run and a later successful re-run), each with a different
run_id and each producing its own repair_attempts row.

i2 and i3 are joined by notebook_execution_id. This is safe for i2 (one
row per id, always - it is the base dataset every attempt resolves
against). It is safe for i3 as long as the given file represents one
coherent explanation run with no duplicate ids; load_i3_index() raises
AmbiguousExplanationError rather than guessing if it finds one.

run_id/created_at prefer the i5 record when one exists (the fix was
actually applied); otherwise they fall back to the i4 record's own
run_id/created_at, since most i4 records (abstained, failed, or simply
not yet run through FixApplicator) never reach i5 at all, and the
architecture's own component diagram still expects those to be logged.
See docs/architecture-note.md section 7.3 for the documented rationale.
"""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from fix_applicator import load_i2_index


DEFAULT_DB_PATH = "data/repair-attempts/repair_attempts.sqlite"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS repair_attempts (
  id                      INTEGER PRIMARY KEY,
  notebook_execution_id   INTEGER NOT NULL,

  failing_module          TEXT,
  subtype                 TEXT,

  explanation             TEXT,

  action                  TEXT,
  install_name            TEXT,
  version                 TEXT,
  command                 TEXT,
  rationale               TEXT,
  pypi_evidence           TEXT,

  outcome                 TEXT,
  new_error_type          TEXT,
  new_error_message       TEXT,

  llm_model               TEXT,
  prompt_strategy         TEXT,
  round                   INTEGER,
  run_id                  TEXT,
  created_at              TEXT
)
"""

REPAIR_ATTEMPT_COLUMNS = [
    "notebook_execution_id",
    "failing_module",
    "subtype",
    "explanation",
    "action",
    "install_name",
    "version",
    "command",
    "rationale",
    "pypi_evidence",
    "outcome",
    "new_error_type",
    "new_error_message",
    "llm_model",
    "prompt_strategy",
    "round",
    "run_id",
    "created_at",
]


class AmbiguousExplanationError(Exception):
    """Raised when an i3 file has more than one explanation record for the
    same notebook_execution_id - there is no way to know which explanation
    a given repair attempt should be logged with, so this refuses to
    guess rather than silently picking one."""


def load_i3_index(path: str) -> Dict[int, Dict[str, Any]]:
    """Load the i3 explanation output keyed by notebook_execution_id."""
    index: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            notebook_execution_id = row.get("input", {}).get("notebook_execution_id")
            if notebook_execution_id is None:
                continue
            notebook_execution_id = int(notebook_execution_id)
            if notebook_execution_id in index:
                raise AmbiguousExplanationError(
                    "duplicate explanation for notebook_execution_id="
                    f"{notebook_execution_id} in {path} (run_ids "
                    f"{index[notebook_execution_id].get('run_id')!r} and "
                    f"{row.get('run_id')!r}) - pass a single-run i3 file"
                )
            index[notebook_execution_id] = row
    return index


def load_i4_records(path: str) -> List[Dict[str, Any]]:
    """Load the i4 repair-proposal output in file order. This is the
    driving set for repair_attempts: each line is one repair attempt."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_i5_index_by_position(path: Optional[str]) -> Dict[int, Dict[str, Any]]:
    """Load the i5 fix-attempt output keyed by its own 'index' field - the
    0-based line number of the i4 file it was run against. Returns an
    empty index if path is None (no fix attempts logged for this batch
    yet)."""
    index: Dict[int, Dict[str, Any]] = {}
    if path is None:
        return index
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            position = row.get("index")
            if position is not None:
                index[int(position)] = row
    return index


def build_repair_attempt_row(
    i4_record: Dict[str, Any],
    i2_row: Dict[str, Any],
    i3_record: Optional[Dict[str, Any]],
    i5_record: Optional[Dict[str, Any]],
    round_number: int = 1,
) -> Dict[str, Any]:
    """Map one i4 record, its required i2 context, and its optional i3/i5
    records onto one repair_attempts row. See module docstring and
    docs/architecture-note.md section 6.2 for the source of each column."""
    explanation_json = None
    llm_model = None
    prompt_strategy = None
    if i3_record is not None:
        explanation_json = i3_record.get("explanation_result", {}).get("explanation_json")
        llm_block = i3_record.get("llm") or {}
        llm_model = llm_block.get("llm_model")
        prompt_strategy = llm_block.get("prompt_strategy")

    if i5_record is not None:
        action = i5_record.get("action")
        install_name = i5_record.get("install_name")
        version = i5_record.get("version")
        command = i5_record.get("command")
        outcome = i5_record.get("outcome")
        new_error_type = i5_record.get("new_error_type")
        new_error_message = i5_record.get("new_error_message")
        run_id = i5_record.get("run_id")
        created_at = i5_record.get("created_at")
    else:
        action = i4_record.get("final_action")
        install_name = i4_record.get("final_install_name")
        version = i4_record.get("final_version")
        command = i4_record.get("command")
        outcome = None
        new_error_type = None
        new_error_message = None
        run_id = i4_record.get("run_id")
        created_at = i4_record.get("created_at")

    retrieval_result = i4_record.get("retrieval_result")

    return {
        "notebook_execution_id": i4_record.get("notebook_execution_id"),
        "failing_module": i2_row.get("failing_module"),
        "subtype": i2_row.get("refined_subtype"),
        "explanation": json.dumps(explanation_json) if explanation_json is not None else None,
        "action": action,
        "install_name": install_name,
        "version": version,
        "command": command,
        "rationale": i4_record.get("final_rationale"),
        "pypi_evidence": json.dumps(retrieval_result) if retrieval_result is not None else None,
        "outcome": outcome,
        "new_error_type": new_error_type,
        "new_error_message": new_error_message,
        "llm_model": llm_model,
        "prompt_strategy": prompt_strategy,
        "round": round_number,
        "run_id": run_id,
        "created_at": created_at,
    }


def create_table(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_TABLE_SQL)
    conn.commit()


def insert_row(conn: sqlite3.Connection, row: Dict[str, Any]) -> int:
    placeholders = ", ".join("?" for _ in REPAIR_ATTEMPT_COLUMNS)
    sql = "INSERT INTO repair_attempts ({}) VALUES ({})".format(
        ", ".join(REPAIR_ATTEMPT_COLUMNS), placeholders
    )
    cursor = conn.execute(sql, [row[column] for column in REPAIR_ATTEMPT_COLUMNS])
    conn.commit()
    return cursor.lastrowid


def log_repair_attempts(
    i2_path: str,
    i4_path: str,
    i3_path: Optional[str],
    i5_path: Optional[str],
    conn: sqlite3.Connection,
    round_number: int = 1,
) -> List[int]:
    """Load all inputs, join, and insert one row per i4 record. Returns
    the inserted row ids in i4 file order. A notebook_execution_id from i4
    that is missing from the i2 dataset is a hard error - i2 is the
    required base context every attempt must resolve against."""
    i2_index = load_i2_index(i2_path)
    i3_index = load_i3_index(i3_path) if i3_path else {}
    i5_index = load_i5_index_by_position(i5_path)
    i4_records = load_i4_records(i4_path)

    create_table(conn)

    inserted_ids = []
    for position, i4_record in enumerate(i4_records):
        notebook_execution_id = i4_record.get("notebook_execution_id")
        if notebook_execution_id is None:
            raise ValueError(
                f"i4 record at index {position} in {i4_path} has no notebook_execution_id"
            )
        notebook_execution_id = int(notebook_execution_id)

        i2_row = i2_index.get(notebook_execution_id)
        if i2_row is None:
            raise ValueError(
                f"notebook_execution_id={notebook_execution_id} (i4 index {position}) "
                f"not found in i2 dataset {i2_path}"
            )

        i3_record = i3_index.get(notebook_execution_id)
        i5_record = i5_index.get(position)

        row = build_repair_attempt_row(i4_record, i2_row, i3_record, i5_record, round_number)
        inserted_ids.append(insert_row(conn, row))

    return inserted_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join i2/i3/i4/i5 outputs into the repair_attempts SQLite table."
    )
    parser.add_argument(
        "--i2", default="data/context-classification/dependency_error_contexts.jsonl"
    )
    parser.add_argument(
        "--i3", default=None, help="i3 explanation JSONL (data/llm-explanations/*.jsonl). Optional."
    )
    parser.add_argument(
        "--i4",
        required=True,
        help="i4 repair-proposal JSONL - the driving file, one repair attempt per line.",
    )
    parser.add_argument(
        "--i5", default=None, help="i5 fix-attempt JSONL (data/fix-attempts/*.jsonl). Optional."
    )
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Drop and recreate the repair_attempts table before logging.",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        if args.overwrite:
            conn.execute("DROP TABLE IF EXISTS repair_attempts")
            conn.commit()
        inserted_ids = log_repair_attempts(
            args.i2, args.i4, args.i3, args.i5, conn, args.round
        )
        print(f"logged {len(inserted_ids)} repair attempt(s) to {db_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
