import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extract_error_contexts import enrich_row, fallback_metadata_only


def memory_connection():
    """An in-memory sqlite3 connection with no `executions` table.

    find_legacy_hint() catches sqlite3.Error and falls back to (None, False),
    so this is a safe, network-free stand-in for the real Docker pipeline DB
    in tests that don't care about legacy-traceback matching.
    """
    return sqlite3.connect(":memory:")


def usable_missing_package_row():
    return {
        "notebook_execution_id": "8",
        "repository_id": "14",
        "notebook_id": "27",
        "notebook_name": "mesure_similarity.ipynb",
        "repository_url": "https://github.com/mdjaffardjy/AnalyseDonneesNextflow",
        "error_type": "ModuleNotFoundError",
        "error_message": "No module named 'sklearn'",
        "error_cell_index": "5",
        "failing_module": "sklearn",
        "subtype": "missing_package",
        "scope_status": "usable",
        "exclusion_reason": "",
        "split": "dev",
    }


def usable_wrong_version_row():
    return {
        "notebook_execution_id": "174",
        "repository_id": "63",
        "notebook_id": "99",
        "notebook_name": "Molten_Salt_Comparison.ipynb",
        "repository_url": "https://github.com/zincware/MDSuite",
        "error_type": "ImportError",
        "error_message": "cannot import name 'cumtrapz' from 'scipy.integrate'",
        "error_cell_index": "1",
        "failing_module": "scipy",
        "subtype": "wrong_version",
        "scope_status": "usable",
        "exclusion_reason": "",
        "split": "dev",
    }


def excluded_system_library_row():
    return {
        "notebook_execution_id": "15",
        "repository_id": "16",
        "notebook_id": "35",
        "notebook_name": "examples/gpax_viDKL_plasmons.ipynb",
        "repository_url": "https://github.com/ziatdinovmax/gpax",
        "error_type": "ImportError",
        "error_message": "libxcb.so.1: cannot open shared object file: No such file or directory",
        "error_cell_index": "8",
        "failing_module": "libxcb.so.1",
        "subtype": "system_library",
        "scope_status": "excluded",
        "exclusion_reason": "requires system library, outside pip-only scope",
        "split": "excluded",
    }


def excluded_mapping_unknown_row():
    return {
        "notebook_execution_id": "999",
        "repository_id": "1",
        "notebook_id": "2",
        "notebook_name": "some_notebook.ipynb",
        "repository_url": "https://github.com/example/example",
        "error_type": "ModuleNotFoundError",
        "error_message": "No module named 'utils'",
        "error_cell_index": "0",
        "failing_module": "utils",
        "subtype": "mapping_unknown",
        "scope_status": "excluded",
        "exclusion_reason": "ambiguous local/module-path import",
        "split": "excluded",
    }


# --- 1. scope fields survive normal enrichment ---------------------------

def test_enrich_row_preserves_scope_fields_for_usable_row():
    row = usable_missing_package_row()

    result = enrich_row(row, memory_connection(), {}, fetch_remote=False)

    assert result["scope_status"] == "usable"
    assert result["exclusion_reason"] == ""
    assert result["split"] == "dev"


def test_enrich_row_preserves_scope_fields_for_excluded_system_library_row():
    row = excluded_system_library_row()

    result = enrich_row(row, memory_connection(), {}, fetch_remote=False)

    assert result["scope_status"] == "excluded"
    assert result["exclusion_reason"] == "requires system library, outside pip-only scope"
    assert result["split"] == "excluded"


def test_enrich_row_preserves_scope_fields_for_excluded_mapping_unknown_row():
    row = excluded_mapping_unknown_row()

    result = enrich_row(row, memory_connection(), {}, fetch_remote=False)

    assert result["scope_status"] == "excluded"
    assert result["exclusion_reason"] == "ambiguous local/module-path import"
    assert result["split"] == "excluded"


# --- 2. scope fields survive the fallback path ----------------------------

def test_fallback_metadata_only_preserves_scope_fields():
    row = excluded_system_library_row()

    result = fallback_metadata_only(row, RuntimeError("simulated enrichment crash"))

    assert result["scope_status"] == "excluded"
    assert result["exclusion_reason"] == "requires system library, outside pip-only scope"
    assert result["split"] == "excluded"


def test_fallback_metadata_only_preserves_scope_fields_for_usable_row():
    row = usable_missing_package_row()

    result = fallback_metadata_only(row, RuntimeError("simulated enrichment crash"))

    assert result["scope_status"] == "usable"
    assert result["split"] == "dev"


# --- 3. excluded rows are not silently treated as repair-eligible ---------

def test_excluded_row_is_distinguishable_from_usable_row():
    excluded = enrich_row(excluded_system_library_row(), memory_connection(), {}, fetch_remote=False)
    usable = enrich_row(usable_missing_package_row(), memory_connection(), {}, fetch_remote=False)

    assert excluded["scope_status"] != usable["scope_status"]
    assert excluded["scope_status"] == "excluded"
    assert excluded["exclusion_reason"]
    assert usable["exclusion_reason"] == ""


# --- 4/6. existing subtype classification and root-cause behavior is unchanged ---

def test_missing_package_row_classification_unchanged():
    result = enrich_row(usable_missing_package_row(), memory_connection(), {}, fetch_remote=False)

    assert result["original_subtype"] == "missing_package"
    assert result["refined_subtype"] == "missing_package"
    assert result["root_cause_hint"] == "import_distribution_name_mismatch"


def test_wrong_version_row_classification_unchanged():
    result = enrich_row(usable_wrong_version_row(), memory_connection(), {}, fetch_remote=False)

    assert result["original_subtype"] == "wrong_version"
    assert result["refined_subtype"] == "wrong_version"
    assert result["root_cause_hint"] == "version_or_api_incompatibility"


def test_system_library_row_classification_unchanged():
    result = enrich_row(excluded_system_library_row(), memory_connection(), {}, fetch_remote=False)

    assert result["refined_subtype"] == "system_library"
    assert result["root_cause_hint"] == "system_level_dependency"
