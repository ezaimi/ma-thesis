import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compatibility_evidence import (
    filter_versions_by_compatibility_evidence,
    load_compatibility_evidence,
    lookup_compatibility_evidence,
)


REGISTRY_PATH = ROOT / "config" / "api_compatibility_evidence.yaml"

SUPPORTED_PATTERNS = [
    ("scipy", "scipy.integrate", "cumtrapz", "<1.14.0"),
    ("scipy", "scipy.sparse.sputils", "isshape", "<1.14.0"),
    ("numpy", "numpy", "VisibleDeprecationWarning", "<2.0.0"),
]


def block_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("compatibility_evidence must never open a network connection")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


# --- 1. every supported dataset pattern resolves --------------------------

@pytest.mark.parametrize("distribution_name,module_path,symbol,expected_specifier", SUPPORTED_PATTERNS)
def test_supported_dataset_pattern_resolves(monkeypatch, distribution_name, module_path, symbol, expected_specifier):
    block_network(monkeypatch)

    result = lookup_compatibility_evidence(distribution_name, module_path, symbol)

    assert result["status"] == "resolved"
    assert result["compatible_specifier"] == expected_specifier
    assert result["evidence"]["source_url"].startswith("https://")


# --- 2/3. exact, case-sensitive module/symbol matching ---------------------

def test_exact_module_and_symbol_matching_works():
    result = lookup_compatibility_evidence("numpy", "numpy", "VisibleDeprecationWarning")
    assert result["status"] == "resolved"


def test_symbol_matching_is_case_sensitive():
    result = lookup_compatibility_evidence("numpy", "numpy", "visibledeprecationwarning")
    assert result["status"] == "no_evidence"


def test_module_path_matching_is_case_sensitive():
    result = lookup_compatibility_evidence("scipy", "Scipy.Integrate", "cumtrapz")
    assert result["status"] == "no_evidence"


def test_distribution_name_is_normalized_not_case_sensitive():
    result = lookup_compatibility_evidence("SciPy", "scipy.integrate", "cumtrapz")
    assert result["status"] == "resolved"


# --- 4. unknown pattern returns no_evidence --------------------------------

def test_unknown_pattern_returns_no_evidence():
    result = lookup_compatibility_evidence("pandas", "pandas", "ix")
    assert result["status"] == "no_evidence"
    assert result["compatible_specifier"] is None
    assert result["error"] is None


# --- 5. invalid registry entries fail safely -------------------------------

def test_entry_missing_required_field_returns_invalid_entry():
    patterns = {
        "foo.bar": {
            "distribution_name": "foo",
            "module_path": "foo",
            "symbol": "bar",
            # missing compatible_specifier and evidence
        }
    }

    result = lookup_compatibility_evidence("foo", "foo", "bar", patterns=patterns)

    assert result["status"] == "invalid_entry"
    assert result["compatible_specifier"] is None
    assert result["error"]


def test_entry_missing_evidence_fields_returns_invalid_entry():
    patterns = {
        "foo.bar": {
            "distribution_name": "foo",
            "module_path": "foo",
            "symbol": "bar",
            "compatible_specifier": "<1.0.0",
            "evidence": {"source_type": "official_release_notes"},  # incomplete
        }
    }

    result = lookup_compatibility_evidence("foo", "foo", "bar", patterns=patterns)

    assert result["status"] == "invalid_entry"


def test_entry_with_non_https_source_url_returns_invalid_entry():
    patterns = {
        "foo.bar": {
            "distribution_name": "foo",
            "module_path": "foo",
            "symbol": "bar",
            "compatible_specifier": "<1.0.0",
            "evidence": {
                "source_type": "official_release_notes",
                "source_url": "http://insecure.example.com/notes",
                "summary": "x",
                "verified_at": "2026-01-01",
            },
        }
    }

    result = lookup_compatibility_evidence("foo", "foo", "bar", patterns=patterns)

    assert result["status"] == "invalid_entry"


def test_malformed_yaml_registry_file_loads_without_crashing(tmp_path):
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("patterns:\n  foo.bar: [not, a, mapping]\n", encoding="utf-8")

    patterns = load_compatibility_evidence(path=str(malformed))
    result = lookup_compatibility_evidence("foo", "foo", "bar", patterns=patterns)

    assert result["status"] == "invalid_entry"


# --- 6. invalid PEP 440 specifier is rejected ------------------------------

def test_invalid_specifier_syntax_returns_invalid_entry():
    patterns = {
        "foo.bar": {
            "distribution_name": "foo",
            "module_path": "foo",
            "symbol": "bar",
            "compatible_specifier": "not-a-real-specifier!!",
            "evidence": {
                "source_type": "official_release_notes",
                "source_url": "https://example.org/notes",
                "summary": "x",
                "verified_at": "2026-01-01",
            },
        }
    }

    result = lookup_compatibility_evidence("foo", "foo", "bar", patterns=patterns)

    assert result["status"] == "invalid_entry"
    assert "specifier" in result["error"].lower()


def test_filter_returns_empty_for_invalid_entry_evidence():
    invalid_evidence = {"status": "invalid_entry", "compatible_specifier": None}

    survivors = filter_versions_by_compatibility_evidence(
        [{"version": "1.0.0"}], invalid_evidence
    )

    assert survivors == []


# --- 7. invalid candidate version is skipped safely ------------------------

def test_unparseable_candidate_version_is_skipped_not_crashed(capsys):
    evidence = lookup_compatibility_evidence("scipy", "scipy.integrate", "cumtrapz")
    candidates = [
        {"version": "not-a-version"},
        {"version": "1.13.1"},
    ]

    survivors = filter_versions_by_compatibility_evidence(candidates, evidence)

    assert [c["version"] for c in survivors] == ["1.13.1"]
    assert "not-a-version" in capsys.readouterr().err


# --- 8/9. constraint-based filtering, not string comparison ----------------

def test_versions_inside_constraint_survive():
    evidence = lookup_compatibility_evidence("scipy", "scipy.integrate", "cumtrapz")
    candidates = [{"version": "1.13.1"}, {"version": "1.9.0"}, {"version": "1.0.0"}]

    survivors = filter_versions_by_compatibility_evidence(candidates, evidence)

    assert {c["version"] for c in survivors} == {"1.13.1", "1.9.0", "1.0.0"}


def test_versions_outside_constraint_are_removed():
    evidence = lookup_compatibility_evidence("scipy", "scipy.integrate", "cumtrapz")
    candidates = [{"version": "1.14.0"}, {"version": "1.18.0"}]

    survivors = filter_versions_by_compatibility_evidence(candidates, evidence)

    assert survivors == []


# --- 10. version ordering does not use string comparison -------------------

def test_filtering_uses_real_version_semantics_not_string_order():
    """String comparison would wrongly treat "1.9.0" as greater than
    "1.14.0" (since '9' > '1' lexically at the second component). A correct
    PEP 440 comparison must still exclude 1.14.0 (>= boundary) while keeping
    1.9.0 (< boundary)."""
    evidence = lookup_compatibility_evidence("scipy", "scipy.integrate", "cumtrapz")
    candidates = [{"version": "1.9.0"}, {"version": "1.14.0"}]

    survivors = filter_versions_by_compatibility_evidence(candidates, evidence)

    assert [c["version"] for c in survivors] == ["1.9.0"]


# --- 11. no command is constructed anywhere in this module ------------------

def test_module_never_builds_a_pip_command():
    import compatibility_evidence

    source = Path(compatibility_evidence.__file__).read_text(encoding="utf-8")

    assert "pip install" not in source
    assert "subprocess" not in source
    assert "os.system" not in source

    evidence = lookup_compatibility_evidence("scipy", "scipy.integrate", "cumtrapz")
    survivors = filter_versions_by_compatibility_evidence([{"version": "1.9.0"}], evidence)

    for candidate in survivors:
        assert "command" not in candidate


# --- 12. no network request is made ----------------------------------------

def test_lookup_and_filter_make_no_network_request(monkeypatch):
    block_network(monkeypatch)

    evidence = lookup_compatibility_evidence("numpy", "numpy", "VisibleDeprecationWarning")
    filter_versions_by_compatibility_evidence([{"version": "1.20.0"}], evidence)
    lookup_compatibility_evidence("unknown", "unknown", "unknown")


# --- 13. every supported entry has an https official source url -----------

def test_every_registry_entry_has_an_https_source_url():
    patterns = load_compatibility_evidence()

    assert len(patterns) >= 3
    for key, entry in patterns.items():
        assert entry["evidence"]["source_url"].startswith("https://"), key


def test_registry_file_covers_all_three_known_dataset_patterns():
    patterns = load_compatibility_evidence(path=str(REGISTRY_PATH))

    assert set(patterns.keys()) == {
        "scipy.integrate.cumtrapz",
        "scipy.sparse.sputils.isshape",
        "numpy.VisibleDeprecationWarning",
    }
