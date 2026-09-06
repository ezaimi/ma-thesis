import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from result_logger import (
    REPAIR_ATTEMPT_COLUMNS,
    AmbiguousExplanationError,
    build_repair_attempt_row,
    create_table,
    insert_row,
    load_i3_index,
    load_i4_records,
    load_i5_index_by_position,
    log_repair_attempts,
)


# --- fixtures modeled on real output shapes (data/context-classification,
# data/llm-explanations, data/repair-proposals) -------------------------------

def i2_row_usable(notebook_execution_id=8, failing_module="sklearn", subtype="missing_package"):
    return {
        "notebook_execution_id": notebook_execution_id,
        "repository_id": 14,
        "notebook_id": 27,
        "failing_module": failing_module,
        "original_subtype": subtype,
        "refined_subtype": subtype,
        "scope_status": "usable",
        "exclusion_reason": None,
    }


def i2_row_excluded(notebook_execution_id=15):
    return {
        "notebook_execution_id": notebook_execution_id,
        "repository_id": 16,
        "notebook_id": 35,
        "failing_module": "libxcb.so.1",
        "original_subtype": "system_library",
        "refined_subtype": "system_library",
        "scope_status": "excluded",
        "exclusion_reason": "requires system library, outside pip-only scope",
    }


def i3_record(notebook_execution_id=8, run_id="i3-20260711T123808Z"):
    return {
        "run_id": run_id,
        "created_at": "2026-07-11T12:39:46.687371+00:00",
        "index": 0,
        "input": {"notebook_execution_id": notebook_execution_id},
        "llm": {
            "llm_model": "gemma2:9b",
            "prompt_strategy": "few_shot",
            "prompt_template": "dependency_explanation_v1",
            "prompt_version": "i3_prompt_v1",
        },
        "explanation_result": {
            "status": "success",
            "explanation_json": {
                "summary": "No module named 'sklearn'.",
                "root_cause": "The notebook imports 'sklearn', which is not installed.",
                "evidence": ["ModuleNotFoundError: No module named 'sklearn'"],
                "failing_module": "sklearn",
                "explanation_confidence": "high",
                "limitations": "Only metadata is available.",
            },
        },
    }


def i4_record_success(
    notebook_execution_id=8,
    run_id="i4-20260817T224854Z",
    action="install",
    install_name="scikit-learn",
    version=None,
    command="python -m pip install scikit-learn",
    rationale="sklearn resolves to the scikit-learn distribution.",
):
    return {
        "run_id": run_id,
        "created_at": "2026-08-17T22:48:54.263357+00:00",
        "notebook_execution_id": notebook_execution_id,
        "status": "success",
        "final_action": action,
        "final_install_name": install_name,
        "final_version": version,
        "final_rationale": rationale,
        "command": command,
        "retrieval_result": {
            "status": "resolved",
            "distribution_name": "scikit-learn",
            "latest_version": "1.9.0",
            "candidate_versions": [
                {"version": "1.7.2", "requires_python": ">=3.10", "python_compatibility": "compatible"},
            ],
            "compatibility_evidence": None,
            "warnings": ["Skipping unparseable PyPI filename: 'scikit-learn-0.9.win32-py2.6.exe'"],
        },
    }


def i4_record_abstained(notebook_execution_id=15, run_id="i4-20260817T224027Z"):
    return {
        "run_id": run_id,
        "created_at": "2026-08-17T22:40:27.718730+00:00",
        "notebook_execution_id": notebook_execution_id,
        "status": "abstained",
        "final_action": "none",
        "final_install_name": None,
        "final_version": None,
        "final_rationale": None,
        "command": None,
        "retrieval_result": None,
    }


def i4_record_failed(notebook_execution_id=8, run_id="i4-20260817T224202Z"):
    return {
        "run_id": run_id,
        "created_at": "2026-08-17T22:42:02.284820+00:00",
        "notebook_execution_id": notebook_execution_id,
        "status": "failed",
        "final_action": "none",
        "final_install_name": None,
        "final_version": None,
        "final_rationale": None,
        "command": None,
        "retrieval_result": {
            "status": "resolved",
            "distribution_name": "scikit-learn",
            "candidate_versions": [],
            "warnings": [],
        },
    }


def i5_record_fixed(index, notebook_execution_id=8, run_id="i5-20260901T000000Z"):
    return {
        "run_id": run_id,
        "created_at": "2026-09-01T00:00:05.000000+00:00",
        "notebook_execution_id": notebook_execution_id,
        "index": index,
        "status": "completed",
        "outcome": "fixed",
        "action": "install",
        "install_name": "scikit-learn",
        "version": None,
        "command": "python -m pip install scikit-learn",
        "new_error_type": None,
        "new_error_message": None,
    }


def i5_record_still_failing(index, notebook_execution_id=174, run_id="i5-20260901T000000Z"):
    return {
        "run_id": run_id,
        "created_at": "2026-09-01T00:00:10.000000+00:00",
        "notebook_execution_id": notebook_execution_id,
        "index": index,
        "status": "completed",
        "outcome": "still_failing",
        "action": "pin_version",
        "install_name": "scipy",
        "version": "1.13.1",
        "command": "python -m pip install scipy==1.13.1",
        "new_error_type": "ImportError",
        "new_error_message": "cannot import name 'foo' from 'scipy'",
    }


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


# --- loader tests --------------------------------------------------------

def test_load_i3_index(tmp_path):
    path = tmp_path / "i3.jsonl"
    write_jsonl(path, [i3_record(notebook_execution_id=8), i3_record(notebook_execution_id=10)])
    index = load_i3_index(str(path))
    assert set(index.keys()) == {8, 10}
    assert index[8]["llm"]["llm_model"] == "gemma2:9b"


def test_load_i3_index_raises_on_duplicate_notebook_execution_id(tmp_path):
    path = tmp_path / "i3.jsonl"
    write_jsonl(
        path,
        [
            i3_record(notebook_execution_id=8, run_id="i3-run-a"),
            i3_record(notebook_execution_id=8, run_id="i3-run-b"),
        ],
    )
    with pytest.raises(AmbiguousExplanationError):
        load_i3_index(str(path))


def test_load_i4_records_preserves_file_order(tmp_path):
    path = tmp_path / "i4.jsonl"
    write_jsonl(
        path,
        [
            i4_record_abstained(notebook_execution_id=15),
            i4_record_failed(notebook_execution_id=8),
            i4_record_success(notebook_execution_id=8),
        ],
    )
    records = load_i4_records(str(path))
    assert [r["notebook_execution_id"] for r in records] == [15, 8, 8]
    assert [r["status"] for r in records] == ["abstained", "failed", "success"]


def test_load_i5_index_by_position_keys_by_index_field(tmp_path):
    path = tmp_path / "i5.jsonl"
    write_jsonl(path, [i5_record_fixed(index=2), i5_record_still_failing(index=0)])
    index = load_i5_index_by_position(str(path))
    assert set(index.keys()) == {0, 2}
    assert index[2]["outcome"] == "fixed"
    assert index[0]["outcome"] == "still_failing"


def test_load_i5_index_by_position_none_path_returns_empty_dict():
    assert load_i5_index_by_position(None) == {}


# --- mapping tests ---------------------------------------------------------

def test_build_row_with_i3_and_i5_present_uses_i5_for_outcome_fields():
    row = build_repair_attempt_row(
        i4_record_success(),
        i2_row_usable(),
        i3_record(),
        i5_record_fixed(index=0),
    )
    assert row["notebook_execution_id"] == 8
    assert row["failing_module"] == "sklearn"
    assert row["subtype"] == "missing_package"
    assert row["action"] == "install"
    assert row["install_name"] == "scikit-learn"
    assert row["version"] is None
    assert row["command"] == "python -m pip install scikit-learn"
    assert row["rationale"] == "sklearn resolves to the scikit-learn distribution."
    assert row["outcome"] == "fixed"
    assert row["new_error_type"] is None
    assert row["run_id"] == "i5-20260901T000000Z"
    assert row["created_at"] == "2026-09-01T00:00:05.000000+00:00"
    assert row["round"] == 1

    explanation = json.loads(row["explanation"])
    assert explanation["summary"] == "No module named 'sklearn'."
    assert explanation["explanation_confidence"] == "high"
    assert row["llm_model"] == "gemma2:9b"
    assert row["prompt_strategy"] == "few_shot"

    evidence = json.loads(row["pypi_evidence"])
    assert evidence["distribution_name"] == "scikit-learn"
    assert evidence["candidate_versions"][0]["version"] == "1.7.2"
    assert "warnings" in evidence


def test_build_row_still_failing_carries_new_error(tmp_path):
    row = build_repair_attempt_row(
        i4_record_success(notebook_execution_id=174, install_name="scipy", version="1.13.1", action="pin_version"),
        i2_row_usable(notebook_execution_id=174, failing_module="scipy", subtype="wrong_version"),
        None,
        i5_record_still_failing(index=3),
    )
    assert row["outcome"] == "still_failing"
    assert row["new_error_type"] == "ImportError"
    assert "foo" in row["new_error_message"]
    assert row["explanation"] is None
    assert row["llm_model"] is None
    assert row["prompt_strategy"] is None


def test_build_row_without_i5_falls_back_to_i4_final_fields_and_run_id():
    row = build_repair_attempt_row(
        i4_record_success(),
        i2_row_usable(),
        i3_record(),
        None,
    )
    assert row["action"] == "install"
    assert row["install_name"] == "scikit-learn"
    assert row["command"] == "python -m pip install scikit-learn"
    assert row["outcome"] is None
    assert row["new_error_type"] is None
    assert row["run_id"] == "i4-20260817T224854Z"
    assert row["created_at"] == "2026-08-17T22:48:54.263357+00:00"


def test_build_row_abstained_excluded_record_has_null_action_and_evidence():
    row = build_repair_attempt_row(
        i4_record_abstained(),
        i2_row_excluded(),
        None,
        None,
    )
    assert row["failing_module"] == "libxcb.so.1"
    assert row["subtype"] == "system_library"
    assert row["action"] == "none"
    assert row["install_name"] is None
    assert row["pypi_evidence"] is None
    assert row["rationale"] is None
    assert row["run_id"] == "i4-20260817T224027Z"


# --- db tests --------------------------------------------------------------

def test_create_table_is_idempotent():
    conn = sqlite3.connect(":memory:")
    create_table(conn)
    create_table(conn)  # must not raise
    cursor = conn.execute("PRAGMA table_info(repair_attempts)")
    columns = {row[1] for row in cursor.fetchall()}
    assert columns == {"id"} | set(REPAIR_ATTEMPT_COLUMNS)


def test_insert_row_round_trips_through_sqlite():
    conn = sqlite3.connect(":memory:")
    create_table(conn)
    row = build_repair_attempt_row(i4_record_success(), i2_row_usable(), i3_record(), i5_record_fixed(index=0))
    row_id = insert_row(conn, row)
    fetched = conn.execute(
        "SELECT {} FROM repair_attempts WHERE id = ?".format(", ".join(REPAIR_ATTEMPT_COLUMNS)),
        (row_id,),
    ).fetchone()
    fetched_row = dict(zip(REPAIR_ATTEMPT_COLUMNS, fetched))
    assert fetched_row["notebook_execution_id"] == 8
    assert fetched_row["outcome"] == "fixed"
    assert json.loads(fetched_row["explanation"])["summary"] == "No module named 'sklearn'."


# --- end-to-end log_repair_attempts tests -----------------------------------

def test_log_repair_attempts_one_row_per_i4_record(tmp_path):
    i2_path = tmp_path / "i2.jsonl"
    i3_path = tmp_path / "i3.jsonl"
    i4_path = tmp_path / "i4.jsonl"
    i5_path = tmp_path / "i5.jsonl"

    write_jsonl(i2_path, [i2_row_usable(notebook_execution_id=8), i2_row_excluded(notebook_execution_id=15)])
    write_jsonl(i3_path, [i3_record(notebook_execution_id=8)])
    write_jsonl(
        i4_path,
        [
            i4_record_abstained(notebook_execution_id=15),  # index 0
            i4_record_success(notebook_execution_id=8),  # index 1
        ],
    )
    write_jsonl(i5_path, [i5_record_fixed(index=1, notebook_execution_id=8)])

    conn = sqlite3.connect(":memory:")
    inserted_ids = log_repair_attempts(str(i2_path), str(i4_path), str(i3_path), str(i5_path), conn)

    assert len(inserted_ids) == 2
    rows = conn.execute(
        "SELECT notebook_execution_id, outcome, explanation FROM repair_attempts ORDER BY id"
    ).fetchall()
    assert rows[0] == (15, None, None)  # abstained, no i3 explanation provided for it either
    assert rows[1][0] == 8
    assert rows[1][1] == "fixed"
    assert json.loads(rows[1][2])["summary"] == "No module named 'sklearn'."


def test_log_repair_attempts_duplicate_notebook_execution_id_across_i4_lines_produces_two_rows(tmp_path):
    """Regression test for the real ambiguity found in
    data/repair-proposals/i4_live_pilot.jsonl: notebook_execution_id=8
    appears twice (a failed attempt, then a successful re-run). Each line
    must become its own repair_attempts row, correctly paired to its own
    i5 outcome by position - not collapsed or cross-matched."""
    i2_path = tmp_path / "i2.jsonl"
    i4_path = tmp_path / "i4.jsonl"
    i5_path = tmp_path / "i5.jsonl"

    write_jsonl(i2_path, [i2_row_usable(notebook_execution_id=8)])
    write_jsonl(
        i4_path,
        [
            i4_record_failed(notebook_execution_id=8, run_id="i4-run-a"),  # index 0
            i4_record_success(notebook_execution_id=8, run_id="i4-run-b"),  # index 1
        ],
    )
    write_jsonl(
        i5_path,
        [
            i5_record_fixed(index=1, notebook_execution_id=8, run_id="i5-run-b"),
        ],
    )

    conn = sqlite3.connect(":memory:")
    inserted_ids = log_repair_attempts(str(i2_path), str(i4_path), None, str(i5_path), conn)

    assert len(inserted_ids) == 2
    rows = conn.execute(
        "SELECT notebook_execution_id, action, outcome, run_id FROM repair_attempts ORDER BY id"
    ).fetchall()
    # first i4 line: failed attempt, never reached i5 -> falls back to i4's own run_id
    assert rows[0] == (8, "none", None, "i4-run-a")
    # second i4 line: the successful re-run, matched to i5 by position (index 1)
    assert rows[1] == (8, "install", "fixed", "i5-run-b")


def test_log_repair_attempts_missing_i2_context_raises(tmp_path):
    i2_path = tmp_path / "i2.jsonl"
    i4_path = tmp_path / "i4.jsonl"
    write_jsonl(i2_path, [])
    write_jsonl(i4_path, [i4_record_success(notebook_execution_id=999)])

    conn = sqlite3.connect(":memory:")
    with pytest.raises(ValueError, match="not found in i2 dataset"):
        log_repair_attempts(str(i2_path), str(i4_path), None, None, conn)


def test_log_repair_attempts_missing_notebook_execution_id_in_i4_raises(tmp_path):
    i2_path = tmp_path / "i2.jsonl"
    i4_path = tmp_path / "i4.jsonl"
    write_jsonl(i2_path, [i2_row_usable()])
    bad_record = i4_record_success()
    del bad_record["notebook_execution_id"]
    write_jsonl(i4_path, [bad_record])

    conn = sqlite3.connect(":memory:")
    with pytest.raises(ValueError, match="no notebook_execution_id"):
        log_repair_attempts(str(i2_path), str(i4_path), None, None, conn)
