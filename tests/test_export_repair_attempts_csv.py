import csv
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from export_repair_attempts_csv import CSV_COLUMNS, export_repair_attempts_csv
from result_logger import REPAIR_ATTEMPT_COLUMNS, create_table


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def i2_row(notebook_execution_id, notebook_id, repository_id=14):
    return {
        "notebook_execution_id": notebook_execution_id,
        "repository_id": repository_id,
        "notebook_id": notebook_id,
        "failing_module": "sklearn",
    }


def insert_repair_attempt(conn, **overrides):
    row = {
        "notebook_execution_id": 8,
        "failing_module": "sklearn",
        "subtype": "missing_package",
        "explanation": json.dumps({"summary": "No module named 'sklearn'."}),
        "action": "install",
        "install_name": "scikit-learn",
        "version": None,
        "command": "python -m pip install scikit-learn",
        "rationale": "sklearn resolves to the scikit-learn distribution.",
        "pypi_evidence": json.dumps({"distribution_name": "scikit-learn", "candidate_versions": [{"version": "1.7.2"}]}),
        "outcome": "fixed",
        "new_error_type": None,
        "new_error_message": None,
        "llm_model": "gemma2:9b",
        "prompt_strategy": "few_shot",
        "round": 1,
        "run_id": "i5-20260901T000000Z",
        "created_at": "2026-09-01T00:00:05.000000+00:00",
    }
    row.update(overrides)
    columns = list(row.keys())
    placeholders = ", ".join("?" for _ in columns)
    cursor = conn.execute(
        f"INSERT INTO repair_attempts ({', '.join(columns)}) VALUES ({placeholders})",
        [row[c] for c in columns],
    )
    conn.commit()
    return cursor.lastrowid


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_csv_columns_are_id_notebook_id_then_repair_attempt_columns():
    assert CSV_COLUMNS[0] == "id"
    assert CSV_COLUMNS[1] == "notebook_id"
    assert CSV_COLUMNS[2:] == REPAIR_ATTEMPT_COLUMNS


def test_export_resolves_notebook_id_from_i2(tmp_path):
    conn = sqlite3.connect(":memory:")
    create_table(conn)
    row_id = insert_repair_attempt(conn, notebook_execution_id=8)

    i2_path = tmp_path / "i2.jsonl"
    write_jsonl(i2_path, [i2_row(notebook_execution_id=8, notebook_id=27, repository_id=14)])

    output_path = tmp_path / "repair_attempts.csv"
    written = export_repair_attempts_csv(conn, str(i2_path), str(output_path))

    assert written == 1
    rows = read_csv_rows(output_path)
    assert len(rows) == 1
    assert rows[0]["id"] == str(row_id)
    assert rows[0]["notebook_id"] == "27"
    assert rows[0]["notebook_execution_id"] == "8"
    assert rows[0]["outcome"] == "fixed"
    assert rows[0]["failing_module"] == "sklearn"


def test_export_header_matches_csv_columns_exactly(tmp_path):
    conn = sqlite3.connect(":memory:")
    create_table(conn)
    insert_repair_attempt(conn)

    i2_path = tmp_path / "i2.jsonl"
    write_jsonl(i2_path, [i2_row(notebook_execution_id=8, notebook_id=27)])

    output_path = tmp_path / "repair_attempts.csv"
    export_repair_attempts_csv(conn, str(i2_path), str(output_path))

    with open(output_path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == CSV_COLUMNS


def test_export_null_columns_become_empty_csv_fields(tmp_path):
    conn = sqlite3.connect(":memory:")
    create_table(conn)
    insert_repair_attempt(
        conn,
        notebook_execution_id=15,
        action="none",
        install_name=None,
        version=None,
        command=None,
        rationale=None,
        pypi_evidence=None,
        outcome=None,
        llm_model=None,
        prompt_strategy=None,
    )

    i2_path = tmp_path / "i2.jsonl"
    write_jsonl(i2_path, [i2_row(notebook_execution_id=15, notebook_id=35, repository_id=16)])

    output_path = tmp_path / "repair_attempts.csv"
    export_repair_attempts_csv(conn, str(i2_path), str(output_path))

    rows = read_csv_rows(output_path)
    assert rows[0]["install_name"] == ""
    assert rows[0]["outcome"] == ""
    assert rows[0]["action"] == "none"


def test_export_preserves_json_blob_through_csv_round_trip(tmp_path):
    conn = sqlite3.connect(":memory:")
    create_table(conn)
    evidence = {
        "distribution_name": "scipy",
        "candidate_versions": [{"version": "1.13.1", "requires_python": ">=3.9"}],
        "compatibility_evidence": {
            "summary": 'release notes say "cumtrapz, removed"',
        },
    }
    insert_repair_attempt(conn, notebook_execution_id=8, pypi_evidence=json.dumps(evidence))

    i2_path = tmp_path / "i2.jsonl"
    write_jsonl(i2_path, [i2_row(notebook_execution_id=8, notebook_id=27)])

    output_path = tmp_path / "repair_attempts.csv"
    export_repair_attempts_csv(conn, str(i2_path), str(output_path))

    rows = read_csv_rows(output_path)
    round_tripped = json.loads(rows[0]["pypi_evidence"])
    assert round_tripped == evidence


def test_export_multiple_rows_ordered_by_id(tmp_path):
    conn = sqlite3.connect(":memory:")
    create_table(conn)
    insert_repair_attempt(conn, notebook_execution_id=15, action="none")
    insert_repair_attempt(conn, notebook_execution_id=8, action="install")

    i2_path = tmp_path / "i2.jsonl"
    write_jsonl(
        i2_path,
        [
            i2_row(notebook_execution_id=15, notebook_id=35, repository_id=16),
            i2_row(notebook_execution_id=8, notebook_id=27, repository_id=14),
        ],
    )

    output_path = tmp_path / "repair_attempts.csv"
    export_repair_attempts_csv(conn, str(i2_path), str(output_path))

    rows = read_csv_rows(output_path)
    assert [r["notebook_execution_id"] for r in rows] == ["15", "8"]
    assert [r["notebook_id"] for r in rows] == ["35", "27"]


def test_export_raises_when_notebook_execution_id_missing_from_i2(tmp_path):
    conn = sqlite3.connect(":memory:")
    create_table(conn)
    insert_repair_attempt(conn, notebook_execution_id=999)

    i2_path = tmp_path / "i2.jsonl"
    write_jsonl(i2_path, [i2_row(notebook_execution_id=8, notebook_id=27)])

    output_path = tmp_path / "repair_attempts.csv"
    with pytest.raises(ValueError, match="not found in i2 dataset"):
        export_repair_attempts_csv(conn, str(i2_path), str(output_path))
