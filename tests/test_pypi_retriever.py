import json
import socket
import sys
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pypi_retriever
from pypi_retriever import (
    DEFAULT_PYTHON_VERSION,
    fetch_pypi_project,
    filter_candidate_versions,
    load_package_mapping,
    normalize_distribution_name,
    resolve_distribution_name,
    retrieve,
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


def test_default_python_version_comes_from_rag_repair_config():
    assert DEFAULT_PYTHON_VERSION == "3.10"


# --- fetch_pypi_project ------------------------------------------------------

class FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_fetch_pypi_project_sends_required_accept_header(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["headers"] = dict(request.header_items())
        return FakeResponse(json.dumps({"files": []}).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    fetch_pypi_project("scikit-learn")

    assert captured["headers"]["Accept"] == "application/vnd.pypi.simple.v1+json"


def test_fetch_pypi_project_returns_files_list_on_success(monkeypatch):
    payload = {"files": [{"filename": "scikit_learn-1.9.0.tar.gz"}]}

    def fake_urlopen(request, timeout=None):
        return FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    status, data = fetch_pypi_project("scikit-learn")

    assert status == "ok"
    assert data["files"] == payload["files"]


def test_fetch_pypi_project_404_returns_package_not_found(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", None, None)

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    status, data = fetch_pypi_project("not-a-real-package-xyz")

    assert status == "package_not_found"
    assert data is None


def test_fetch_pypi_project_connection_failure_returns_network_error(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    status, data = fetch_pypi_project("scikit-learn")

    assert status == "network_error"
    assert data is None


def test_fetch_pypi_project_timeout_returns_network_error(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise socket.timeout("timed out")

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    status, data = fetch_pypi_project("scikit-learn")

    assert status == "network_error"


def test_fetch_pypi_project_malformed_json_returns_invalid_response(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return FakeResponse(b"not json{{{")

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    status, data = fetch_pypi_project("scikit-learn")

    assert status == "invalid_response"
    assert data is None


def test_fetch_pypi_project_missing_files_key_returns_invalid_response(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return FakeResponse(json.dumps({"name": "scikit-learn"}).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    status, data = fetch_pypi_project("scikit-learn")

    assert status == "invalid_response"


def test_fetch_pypi_project_non_list_files_returns_invalid_response(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return FakeResponse(json.dumps({"files": "not-a-list"}).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    status, data = fetch_pypi_project("scikit-learn")

    assert status == "invalid_response"


# --- filter_candidate_versions -----------------------------------------------

def load_l5_poc_fixture_as_files():
    fixture_path = ROOT / "data" / "prompt-tests" / "l5_pypi_poc_filter_fixture.json"
    with fixture_path.open(encoding="utf-8") as f:
        fixture = json.load(f)

    files = []
    for entry in fixture["input_files"]:
        file_dict = {
            "filename": f"examplepkg-{entry['version']}.tar.gz",
            "yanked": entry["yanked"],
        }
        if entry["requires_python"] is not None:
            file_dict["requires-python"] = entry["requires_python"]
        files.append(file_dict)

    return files, fixture


def test_filter_candidate_versions_matches_l5_poc_fixture_exactly():
    files, fixture = load_l5_poc_fixture_as_files()

    result = filter_candidate_versions(files, fixture["python_version"], limit=5)

    assert result == fixture["candidate_versions"]


def test_filter_candidate_versions_excludes_fully_yanked_release():
    files, _ = load_l5_poc_fixture_as_files()

    result = filter_candidate_versions(files, "3.13", limit=5)

    assert "2.5.0" not in [c["version"] for c in result]


def test_filter_candidate_versions_excludes_prerelease():
    files, _ = load_l5_poc_fixture_as_files()

    result = filter_candidate_versions(files, "3.13", limit=5)

    assert "1.9.0b1" not in [c["version"] for c in result]


def test_filter_candidate_versions_excludes_incompatible_release():
    files, _ = load_l5_poc_fixture_as_files()

    result = filter_candidate_versions(files, "3.13", limit=5)

    assert "2.0.0" not in [c["version"] for c in result]


def test_filter_candidate_versions_retains_unknown_compatibility_release():
    """A release with no declared requires-python is kept, not dropped -
    marked unknown, never claimed compatible. Matches the L5 PoC fixture,
    not day-1-rag-repair-agent-plan.md's stricter prose (superseded, see
    docs/rag-design.md §2.2/§7)."""
    files, _ = load_l5_poc_fixture_as_files()

    result = filter_candidate_versions(files, "3.13", limit=5)
    unknown = next(c for c in result if c["version"] == "1.8.0")

    assert unknown["python_compatibility"] == "unknown"


def test_filter_candidate_versions_with_unknown_python_version_marks_all_unknown():
    files = [
        {"filename": "pkg-1.0.0.tar.gz", "yanked": False, "requires-python": ">=3.9"},
        {"filename": "pkg-2.0.0.tar.gz", "yanked": False, "requires-python": ">=3.9"},
    ]

    result = filter_candidate_versions(files, None, limit=5)

    assert all(c["python_compatibility"] == "unknown" for c in result)
    assert {c["version"] for c in result} == {"1.0.0", "2.0.0"}


def test_filter_candidate_versions_caps_at_limit():
    files = [
        {"filename": f"pkg-{i}.0.0.tar.gz", "yanked": False}
        for i in range(1, 9)
    ]

    result = filter_candidate_versions(files, None, limit=5)

    assert len(result) == 5


def test_filter_candidate_versions_sorts_by_real_version_not_string_order():
    """String comparison would put "1.10.0" before "1.9.0". Real version
    comparison must not."""
    files = [
        {"filename": "pkg-1.9.0.tar.gz", "yanked": False},
        {"filename": "pkg-1.10.0.tar.gz", "yanked": False},
    ]

    result = filter_candidate_versions(files, None, limit=5)

    assert [c["version"] for c in result] == ["1.10.0", "1.9.0"]


def test_filter_candidate_versions_skips_unparseable_filename_safely(capsys):
    files = [
        {"filename": "not-a-valid-filename", "yanked": False},
        {"filename": "pkg-1.0.0.tar.gz", "yanked": False},
    ]

    result = filter_candidate_versions(files, None, limit=5)

    assert [c["version"] for c in result] == ["1.0.0"]
    assert "not-a-valid-filename" in capsys.readouterr().err


# --- retrieve() ---------------------------------------------------------------

def test_retrieve_mapping_unknown_makes_no_network_request(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("retrieve() must not query PyPI for an unresolved import name")

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fail_if_called)

    result = retrieve("dms_variants")

    assert result["status"] == "mapping_unknown"
    assert result["distribution_name"] is None
    assert result["package_found"] is None
    assert result["candidate_versions"] == []
    assert result["source_endpoint"] is None


def test_retrieve_uses_configured_python_version_by_default(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return FakeResponse(json.dumps({"files": []}).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    result = retrieve("sklearn")

    assert result["python_version"] == "3.10"


def test_retrieve_accepts_an_explicit_python_version_override(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return FakeResponse(json.dumps({"files": []}).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    result = retrieve("sklearn", python_version="3.11")

    assert result["python_version"] == "3.11"


def test_retrieve_resolved_path_returns_full_schema(monkeypatch):
    payload = {
        "files": [
            {"filename": "scikit_learn-1.9.0.tar.gz", "yanked": False, "requires-python": ">=3.10"},
            {"filename": "scikit_learn-1.8.0.tar.gz", "yanked": False, "requires-python": ">=3.10"},
        ]
    }

    def fake_urlopen(request, timeout=None):
        assert "scikit-learn" in request.full_url
        return FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    result = retrieve("sklearn", python_version="3.10")

    assert result["status"] == "resolved"
    assert result["distribution_name"] == "scikit-learn"
    assert result["package_found"] is True
    assert result["latest_version"] == "1.9.0"
    assert [c["version"] for c in result["candidate_versions"]] == ["1.9.0", "1.8.0"]
    assert result["source_endpoint"] == "https://pypi.org/simple/scikit-learn/"
    assert result["retrieved_at"] is not None
    assert result["error"] is None
    assert set(result.keys()) == {
        "status", "import_name", "distribution_name", "package_found",
        "python_version", "latest_version", "candidate_versions",
        "source_endpoint", "retrieved_at", "error",
    }


def test_retrieve_package_not_found(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", None, None)

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    result = retrieve("sklearn")

    assert result["status"] == "package_not_found"
    assert result["package_found"] is False
    assert result["candidate_versions"] == []


def test_retrieve_network_error(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    result = retrieve("sklearn")

    assert result["status"] == "network_error"
    assert result["error"]
    assert result["candidate_versions"] == []


def test_retrieve_invalid_response(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return FakeResponse(b"not json{{{")

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    result = retrieve("sklearn")

    assert result["status"] == "invalid_response"
    assert result["error"]


def test_retrieve_no_compatible_release_when_everything_is_incompatible(monkeypatch):
    payload = {
        "files": [
            {"filename": "scikit_learn-1.0.0.tar.gz", "yanked": False, "requires-python": "<3.7"},
        ]
    }

    def fake_urlopen(request, timeout=None):
        return FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    result = retrieve("sklearn", python_version="3.10")

    assert result["status"] == "no_compatible_release"
    assert result["candidate_versions"] == []


def test_retrieve_never_returns_a_command_field(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return FakeResponse(json.dumps({
            "files": [{"filename": "scikit_learn-1.9.0.tar.gz", "yanked": False}]
        }).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    result = retrieve("sklearn")

    assert "command" not in result
    for candidate in result["candidate_versions"]:
        assert "command" not in candidate


def test_module_never_builds_a_pip_command():
    source = Path(pypi_retriever.__file__).read_text(encoding="utf-8")

    assert "pip install" not in source
    assert "subprocess" not in source
    assert "os.system" not in source
