import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pypi_retriever import (
    load_package_mapping,
    normalize_distribution_name,
    resolve_distribution_name,
)


def test_resolve_known_import_returns_verified_distribution():
    assert resolve_distribution_name("sklearn") == "scikit-learn"


def test_resolve_is_case_sensitive():
    assert resolve_distribution_name("Bio") == "biopython"
    assert resolve_distribution_name("bio") is None


def test_resolve_unknown_import_returns_none():
    assert resolve_distribution_name("dms_variants") is None


def test_resolve_does_not_assume_identity_mapping():
    assert resolve_distribution_name("not_a_real_import_xyz") is None


def test_normalize_replaces_underscore_with_hyphen():
    assert normalize_distribution_name("scikit_learn") == "scikit-learn"


def test_normalize_lowercases_and_collapses_separators():
    assert normalize_distribution_name("My.Package_Name") == "my-package-name"


def test_normalize_collapses_consecutive_separators():
    assert normalize_distribution_name("foo___bar..baz") == "foo-bar-baz"


def test_package_mapping_loads_expected_entries():
    mapping = load_package_mapping()

    assert mapping["sklearn"] == "scikit-learn"
    assert mapping["umap"] == "umap-learn"
    assert mapping["pkg_resources"] == "setuptools"
    assert mapping["Bio"] == "biopython"
    assert mapping["scipy"] == "scipy"
    assert mapping["numpy"] == "numpy"
    assert mapping["pandas"] == "pandas"
