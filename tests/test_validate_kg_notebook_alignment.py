import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_kg_notebook_alignment import (
    DuplicateSourceIdError,
    check_record,
    load_csv_by_id,
    main,
    normalize_repository_url,
    summarize,
    validate_alignment,
)


# --- fixtures modeled on the real i2 dataset and the real external FAIR
# Jupyter KG's repositories.csv/notebooks.csv (verified by hand against
# repository_id=14/notebook_id=27 in the real checkout) ----------------------

def i2_row(
    notebook_execution_id=8,
    repository_id=14,
    notebook_id=27,
    repository_url="https://github.com/mdjaffardjy/AnalyseDonneesNextflow",
    notebook_name="Analysis/Similarity Processes/mesure_similarity.ipynb",
):
    return {
        "notebook_execution_id": notebook_execution_id,
        "repository_id": repository_id,
        "notebook_id": notebook_id,
        "repository_url": repository_url,
        "notebook_name": notebook_name,
    }


def repositories_csv_row(id=14, domain="github.com", repository="mdjaffardjy/AnalyseDonneesNextflow"):
    return {"id": id, "domain": domain, "repository": repository}


def notebooks_csv_row(id=27, repository_id=14, name="Analysis/Similarity Processes/mesure_similarity.ipynb"):
    return {"id": id, "repository_id": repository_id, "name": name}


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


# --- normalize_repository_url ------------------------------------------------

def test_normalize_repository_url_builds_https_github_url():
    assert normalize_repository_url("github.com", "owner/repo") == "https://github.com/owner/repo"


def test_normalize_repository_url_strips_trailing_slash():
    assert normalize_repository_url("github.com", "owner/repo/") == "https://github.com/owner/repo"


# --- load_csv_by_id, including malformed source data -------------------------

def test_load_csv_by_id_keys_by_int_id(tmp_path):
    path = tmp_path / "repositories.csv"
    write_csv(path, ["id", "domain", "repository"], [repositories_csv_row(id=14)])
    index = load_csv_by_id(str(path))
    assert set(index.keys()) == {14}
    assert index[14]["repository"] == "mdjaffardjy/AnalyseDonneesNextflow"


def test_load_csv_by_id_raises_on_duplicate_id(tmp_path):
    """Regression guard: two rows sharing the same id in the external KG's
    own CSV is malformed reference data, not something to silently resolve
    by picking the last row seen."""
    path = tmp_path / "repositories.csv"
    write_csv(
        path,
        ["id", "domain", "repository"],
        [repositories_csv_row(id=14, repository="a/a"), repositories_csv_row(id=14, repository="b/b")],
    )
    with pytest.raises(DuplicateSourceIdError):
        load_csv_by_id(str(path))


# --- check_record: the required per-record scenarios --------------------------

def test_check_record_complete_successful_match():
    repositories_index = {14: repositories_csv_row()}
    notebooks_index = {27: notebooks_csv_row()}
    result = check_record(i2_row(), repositories_index, notebooks_index)
    assert result["repository_status"] == "match"
    assert result["notebook_status"] == "match"
    assert result["detail"] is None


def test_check_record_repository_mismatch():
    repositories_index = {14: repositories_csv_row(repository="someone-else/other-repo")}
    notebooks_index = {27: notebooks_csv_row()}
    result = check_record(i2_row(), repositories_index, notebooks_index)
    assert result["repository_status"] == "mismatch"
    assert "repository" in result["detail"]


def test_check_record_notebook_mismatch_on_name():
    repositories_index = {14: repositories_csv_row()}
    notebooks_index = {27: notebooks_csv_row(name="different_notebook.ipynb")}
    result = check_record(i2_row(), repositories_index, notebooks_index)
    assert result["repository_status"] == "match"
    assert result["notebook_status"] == "mismatch"
    assert "name" in result["detail"]


def test_check_record_notebook_mismatch_on_cross_repository_linkage():
    """A notebook_id that exists in notebooks.csv and even has the right
    name, but is recorded there under a different repository_id, must not
    count as a match - that would be a coincidental id collision across
    repositories, not the same notebook."""
    repositories_index = {14: repositories_csv_row()}
    notebooks_index = {27: notebooks_csv_row(repository_id=999)}
    result = check_record(i2_row(), repositories_index, notebooks_index)
    assert result["notebook_status"] == "mismatch"
    assert "repository_id" in result["detail"]


def test_check_record_missing_repository():
    repositories_index = {}
    notebooks_index = {27: notebooks_csv_row()}
    result = check_record(i2_row(), repositories_index, notebooks_index)
    assert result["repository_status"] == "missing"
    assert result["notebook_status"] == "match"


def test_check_record_missing_notebook():
    repositories_index = {14: repositories_csv_row()}
    notebooks_index = {}
    result = check_record(i2_row(), repositories_index, notebooks_index)
    assert result["repository_status"] == "match"
    assert result["notebook_status"] == "missing"


# --- validate_alignment / summarize --------------------------------------------

def test_validate_alignment_and_summarize_all_match():
    rows = [
        i2_row(notebook_execution_id=8),
        i2_row(notebook_execution_id=9, repository_id=14, notebook_id=27),
    ]
    repositories_index = {14: repositories_csv_row()}
    notebooks_index = {27: notebooks_csv_row()}
    results = validate_alignment(rows, repositories_index, notebooks_index)
    summary = summarize(results)
    assert summary == {
        "total": 2,
        "repository_matches": 2,
        "notebook_matches": 2,
        "missing": 0,
        "mismatches": 0,
    }


def test_summarize_counts_missing_separately_from_mismatches():
    rows = [
        i2_row(notebook_execution_id=8, repository_id=14, notebook_id=27),
        i2_row(notebook_execution_id=9, repository_id=999, notebook_id=27),  # missing repository
        i2_row(notebook_execution_id=10, repository_id=14, notebook_id=27, notebook_name="wrong.ipynb"),  # mismatch
    ]
    repositories_index = {14: repositories_csv_row()}
    notebooks_index = {27: notebooks_csv_row()}
    results = validate_alignment(rows, repositories_index, notebooks_index)
    summary = summarize(results)
    assert summary["total"] == 3
    assert summary["missing"] == 1
    assert summary["mismatches"] == 1


# --- main() end-to-end CLI ------------------------------------------------------

def test_main_exits_zero_and_reports_success_on_full_match(tmp_path, capsys, monkeypatch):
    i2_path = tmp_path / "i2.jsonl"
    write_jsonl(i2_path, [i2_row()])

    repositories_path = tmp_path / "repositories.csv"
    write_csv(repositories_path, ["id", "domain", "repository"], [repositories_csv_row()])

    notebooks_path = tmp_path / "notebooks.csv"
    write_csv(notebooks_path, ["id", "repository_id", "name"], [notebooks_csv_row()])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_kg_notebook_alignment.py",
            "--i2", str(i2_path),
            "--repositories-csv", str(repositories_path),
            "--notebooks-csv", str(notebooks_path),
        ],
    )
    main()  # must not raise / must not sys.exit(1)
    out = capsys.readouterr().out
    assert "Records checked: 1" in out
    assert "Repository matches: 1/1" in out
    assert "Notebook matches: 1/1" in out
    assert "Missing: 0" in out
    assert "Mismatches: 0" in out


def test_main_exits_nonzero_on_mismatch(tmp_path, monkeypatch):
    i2_path = tmp_path / "i2.jsonl"
    write_jsonl(i2_path, [i2_row(notebook_name="wrong.ipynb")])

    repositories_path = tmp_path / "repositories.csv"
    write_csv(repositories_path, ["id", "domain", "repository"], [repositories_csv_row()])

    notebooks_path = tmp_path / "notebooks.csv"
    write_csv(notebooks_path, ["id", "repository_id", "name"], [notebooks_csv_row()])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_kg_notebook_alignment.py",
            "--i2", str(i2_path),
            "--repositories-csv", str(repositories_path),
            "--notebooks-csv", str(notebooks_path),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
