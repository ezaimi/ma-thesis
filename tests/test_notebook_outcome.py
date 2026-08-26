import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from notebook_outcome import (
    NotebookReadError,
    classify_outcome,
    find_first_error_output,
    is_same_dependency_error,
    load_output_notebook,
)


def _notebook(cells):
    return {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def _code_cell(outputs):
    return {"cell_type": "code", "source": "", "outputs": outputs}


def _error_output(ename, evalue):
    return {"output_type": "error", "ename": ename, "evalue": evalue, "traceback": []}


# --- load_output_notebook -----------------------------------------------


def test_load_output_notebook_missing_file_raises(tmp_path):
    with pytest.raises(NotebookReadError):
        load_output_notebook(tmp_path / "does_not_exist.ipynb")


def test_load_output_notebook_malformed_json_raises(tmp_path):
    path = tmp_path / "broken.ipynb"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(NotebookReadError):
        load_output_notebook(path)


def test_load_output_notebook_missing_cells_key_raises(tmp_path):
    path = tmp_path / "not_a_notebook.ipynb"
    path.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    with pytest.raises(NotebookReadError):
        load_output_notebook(path)


def test_load_output_notebook_valid_file_loads(tmp_path):
    path = tmp_path / "ok.ipynb"
    path.write_text(json.dumps(_notebook([_code_cell([])])), encoding="utf-8")
    notebook = load_output_notebook(path)
    assert notebook["cells"] == [_code_cell([])]


# --- find_first_error_output ----------------------------------------------


def test_find_first_error_output_none_when_clean():
    notebook = _notebook([_code_cell([]), _code_cell([{"output_type": "stream", "text": "hi"}])])
    assert find_first_error_output(notebook) is None


def test_find_first_error_output_finds_first_error():
    notebook = _notebook(
        [
            _code_cell([]),
            _code_cell([_error_output("ModuleNotFoundError", "No module named 'sklearn'")]),
            _code_cell([_error_output("ValueError", "should never be reached")]),
        ]
    )
    error = find_first_error_output(notebook)
    assert error == {"ename": "ModuleNotFoundError", "evalue": "No module named 'sklearn'"}


def test_find_first_error_output_skips_malformed_cells_and_outputs():
    notebook = _notebook(
        [
            "not a dict",
            {"cell_type": "code", "outputs": "not a list"},
            _code_cell(["not a dict output", {"output_type": "stream"}]),
        ]
    )
    assert find_first_error_output(notebook) is None


# --- is_same_dependency_error ----------------------------------------------


def test_is_same_dependency_error_true_for_matching_type_and_module():
    assert is_same_dependency_error(
        "ModuleNotFoundError", "No module named 'sklearn'", "ModuleNotFoundError", "sklearn"
    )


def test_is_same_dependency_error_false_for_different_type():
    assert not is_same_dependency_error(
        "ImportError", "No module named 'sklearn'", "ModuleNotFoundError", "sklearn"
    )


def test_is_same_dependency_error_false_for_different_module():
    assert not is_same_dependency_error(
        "ModuleNotFoundError", "No module named 'numpy'", "ModuleNotFoundError", "sklearn"
    )


def test_is_same_dependency_error_false_when_original_fields_missing():
    assert not is_same_dependency_error("ModuleNotFoundError", "No module named 'sklearn'", None, "sklearn")
    assert not is_same_dependency_error("ModuleNotFoundError", "No module named 'sklearn'", "ModuleNotFoundError", None)


# --- classify_outcome --------------------------------------------------------


def test_classify_outcome_fixed_when_no_error_cell():
    notebook = _notebook([_code_cell([]), _code_cell([{"output_type": "stream", "text": "done"}])])
    result = classify_outcome(notebook, "ModuleNotFoundError", "sklearn")
    assert result == {
        "outcome": "fixed",
        "new_error_type": None,
        "new_error_message": None,
        "same_as_original_error": False,
    }


def test_classify_outcome_still_failing_same_error():
    notebook = _notebook([_code_cell([_error_output("ModuleNotFoundError", "No module named 'sklearn'")])])
    result = classify_outcome(notebook, "ModuleNotFoundError", "sklearn")
    assert result["outcome"] == "still_failing"
    assert result["new_error_type"] == "ModuleNotFoundError"
    assert result["new_error_message"] == "No module named 'sklearn'"
    assert result["same_as_original_error"] is True


def test_classify_outcome_still_failing_different_error():
    notebook = _notebook([_code_cell([_error_output("NameError", "name 'foo' is not defined")])])
    result = classify_outcome(notebook, "ModuleNotFoundError", "sklearn")
    assert result["outcome"] == "still_failing"
    assert result["new_error_type"] == "NameError"
    assert result["same_as_original_error"] is False
