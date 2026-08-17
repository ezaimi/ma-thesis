import json
import socket
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pypi_retriever
from pypi_retriever import (
    DEFAULT_PYTHON_VERSION,
    fetch_pypi_project,
    filter_candidate_versions,
    load_package_mapping,
    load_rag_repair_config,
    normalize_distribution_name,
    resolve_distribution_name,
    retrieve,
    validate_python_version_config,
)


@pytest.fixture(autouse=True)
def _reset_pypi_cache(monkeypatch):
    """The in-memory fetch cache and the rate-limit clock are both
    process-global; without resetting them, an earlier test's mocked
    response or throttling state would leak into a later, unrelated test.

    Also monkeypatches time.sleep to a no-op by default, so a test that
    doesn't care about rate limiting (e.g. one making two real,
    non-monkeypatched requests) never actually blocks for real - a test
    that specifically wants to observe sleeping overrides this itself with
    its own monkeypatch.setattr(pypi_retriever.time, "sleep", ...) later in
    the same test, which simply layers on top.
    """
    pypi_retriever.clear_pypi_cache()
    pypi_retriever.reset_pypi_rate_limiter()
    monkeypatch.setattr(pypi_retriever.time, "sleep", lambda seconds: None)
    yield
    pypi_retriever.clear_pypi_cache()
    pypi_retriever.reset_pypi_rate_limiter()


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


# --- repair-runtime configuration (config/rag_repair.yaml) ------------------

def test_default_python_version_comes_from_rag_repair_config():
    assert DEFAULT_PYTHON_VERSION == "3.10"


def test_load_rag_repair_config_reads_python_version():
    config = load_rag_repair_config()
    assert config["runtime"]["python_version"] == "3.10"


def test_load_rag_repair_config_accepts_injected_path(tmp_path):
    config_path = tmp_path / "custom_rag_repair.yaml"
    config_path.write_text("runtime:\n  python_version: \"3.12\"\n", encoding="utf-8")

    config = load_rag_repair_config(path=str(config_path))

    assert config["runtime"]["python_version"] == "3.12"


def test_load_rag_repair_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rag_repair_config(path=str(tmp_path / "does_not_exist.yaml"))


def test_validate_python_version_config_accepts_valid_version():
    version, error = validate_python_version_config("3.10")
    assert version == "3.10"
    assert error is None


def test_validate_python_version_config_rejects_missing_value():
    version, error = validate_python_version_config(None)
    assert version is None
    assert "missing" in error.lower()


def test_validate_python_version_config_rejects_empty_string():
    version, error = validate_python_version_config("")
    assert version is None
    assert error is not None


def test_validate_python_version_config_rejects_non_string():
    version, error = validate_python_version_config(310)
    assert version is None
    assert error is not None


def test_validate_python_version_config_rejects_malformed_version():
    version, error = validate_python_version_config("not-a-version")
    assert version is None
    assert "not a valid" in error.lower()


def test_retrieve_uses_configuration_error_when_default_config_is_broken(monkeypatch):
    """A malformed/missing config must never silently fall back to notebook
    metadata or a second hardcoded version - it must fail as a structured
    retrieval status, with no PyPI request attempted."""
    monkeypatch.setattr(pypi_retriever, "DEFAULT_PYTHON_VERSION", None)
    monkeypatch.setattr(pypi_retriever, "DEFAULT_PYTHON_VERSION_ERROR", "simulated broken config")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("must not query PyPI when the default Python version is unresolvable")

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fail_if_called)

    result = retrieve("sklearn")

    assert result["status"] == "configuration_error"
    assert result["error"] == "simulated broken config"


def test_retrieve_explicit_python_version_bypasses_broken_default_config(monkeypatch):
    """An explicit override is dependency injection for callers/tests, not
    a silent alternate production source - it works even if the on-disk
    default config is broken, because the caller supplied a value directly
    instead of relying on the file."""
    monkeypatch.setattr(pypi_retriever, "DEFAULT_PYTHON_VERSION", None)
    monkeypatch.setattr(pypi_retriever, "DEFAULT_PYTHON_VERSION_ERROR", "simulated broken config")

    def fake_urlopen(request, timeout=None):
        return FakeResponse(json.dumps({"files": []}).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    result = retrieve("sklearn", python_version="3.11")

    assert result["status"] != "configuration_error"
    assert result["python_version"] == "3.11"


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


def test_fetch_pypi_project_sends_required_accept_and_user_agent_headers(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["headers"] = dict(request.header_items())
        captured["url"] = request.full_url
        return FakeResponse(json.dumps({"files": []}).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    fetch_pypi_project("scikit-learn")

    assert captured["headers"]["Accept"] == "application/vnd.pypi.simple.v1+json"
    assert captured["headers"]["User-agent"] == pypi_retriever.USER_AGENT
    assert captured["url"] == "https://pypi.org/simple/scikit-learn/"


def test_fetch_pypi_project_only_targets_the_official_pypi_host(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        return FakeResponse(json.dumps({"files": []}).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    fetch_pypi_project("umap-learn")

    assert captured["url"].startswith("https://pypi.org/simple/")


def test_fetch_pypi_project_rejects_unnormalized_name_without_a_request(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("must not build a URL from an unnormalized/unsafe name")

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fail_if_called)

    status, data = fetch_pypi_project("Not Normalized/../etc")

    assert status == "invalid_response"
    assert data is None


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


def test_fetch_pypi_project_other_http_error_returns_network_error(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 503, "Service Unavailable", None, None)

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    status, data = fetch_pypi_project("scikit-learn")

    assert status == "network_error"


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


def test_fetch_pypi_project_accepts_injected_urlopen_without_monkeypatching_module():
    calls = {"count": 0}

    def fake_urlopen(request, timeout=None):
        calls["count"] += 1
        return FakeResponse(json.dumps({"files": []}).encode("utf-8"))

    status, data = fetch_pypi_project("scikit-learn", urlopen=fake_urlopen)

    assert status == "ok"
    assert calls["count"] == 1


def test_fetch_pypi_project_caches_repeated_requests_within_a_run(monkeypatch):
    calls = {"count": 0}

    def fake_urlopen(request, timeout=None):
        calls["count"] += 1
        return FakeResponse(json.dumps({"files": []}).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    fetch_pypi_project("scikit-learn")
    fetch_pypi_project("scikit-learn")
    fetch_pypi_project("scikit-learn")

    assert calls["count"] == 1


def test_fetch_pypi_project_use_cache_false_bypasses_cache(monkeypatch):
    calls = {"count": 0}

    def fake_urlopen(request, timeout=None):
        calls["count"] += 1
        return FakeResponse(json.dumps({"files": []}).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    fetch_pypi_project("scikit-learn", use_cache=False)
    fetch_pypi_project("scikit-learn", use_cache=False)

    assert calls["count"] == 2


def test_retrieve_does_not_repeat_a_pypi_request_for_the_same_distribution(monkeypatch):
    calls = {"count": 0}

    def fake_urlopen(request, timeout=None):
        calls["count"] += 1
        return FakeResponse(json.dumps({
            "files": [{"filename": "scikit_learn-1.9.0.tar.gz", "yanked": False}]
        }).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    retrieve("sklearn")
    retrieve("sklearn")

    assert calls["count"] == 1


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


def test_unknown_and_compatible_candidates_are_not_schema_distinguishable_beyond_the_tag():
    """"unknown" and "compatible" entries sit in the same candidate_versions
    list, distinguished only by python_compatibility - a caller that reads
    "version" without also checking python_compatibility could mistake an
    unproven release for a confirmed one. Documented as a known risk (see
    the audit report); not fixed by data-shape change in this task."""
    files, _ = load_l5_poc_fixture_as_files()

    result = filter_candidate_versions(files, "3.13", limit=5)
    tags = {c["python_compatibility"] for c in result}

    assert "compatible" in tags
    assert "unknown" in tags
    assert all("python_compatibility" in c for c in result)


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


def test_filter_candidate_versions_limit_none_returns_everything_uncapped():
    files = [
        {"filename": f"pkg-{i}.0.0.tar.gz", "yanked": False}
        for i in range(1, 9)
    ]

    result = filter_candidate_versions(files, None, limit=None)

    assert len(result) == 8


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


def test_filter_candidate_versions_skips_malformed_requires_python(capsys):
    files = [
        {"filename": "pkg-1.0.0.tar.gz", "yanked": False, "requires-python": "not a specifier!!"},
    ]

    result = filter_candidate_versions(files, "3.10", limit=5)

    assert result[0]["python_compatibility"] == "unknown"
    assert "invalid requires-python" in capsys.readouterr().err.lower()


def test_filter_candidate_versions_wheel_and_sdist_for_same_version_produce_one_candidate():
    files = [
        {"filename": "scikit_learn-1.9.0-cp310-cp310-manylinux_2_17_x86_64.whl", "yanked": False},
        {"filename": "scikit-learn-1.9.0.tar.gz", "yanked": False},
    ]

    result = filter_candidate_versions(files, None, limit=5)

    assert len(result) == 1
    assert result[0]["version"] == "1.9.0"


# --- retrieve(): missing_package ---------------------------------------------

def test_retrieve_mapping_unknown_makes_no_network_request(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("retrieve() must not query PyPI for an unresolved import name")

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fail_if_called)

    result = retrieve("dms_variants")

    assert result["status"] == "mapping_unknown"
    assert result["subtype"] == "missing_package"
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
    assert result["subtype"] == "missing_package"
    assert result["distribution_name"] == "scikit-learn"
    assert result["package_found"] is True
    assert result["latest_version"] == "1.9.0"
    assert [c["version"] for c in result["candidate_versions"]] == ["1.9.0", "1.8.0"]
    assert result["source_endpoint"] == "https://pypi.org/simple/scikit-learn/"
    assert result["retrieved_at"] is not None
    assert result["compatibility_evidence"] is None
    assert result["warnings"] == []
    assert result["error"] is None
    assert set(result.keys()) == {
        "status", "import_name", "subtype", "module_path", "symbol",
        "distribution_name", "package_found", "python_version",
        "latest_version", "candidate_versions", "compatibility_evidence",
        "source_endpoint", "retrieved_at", "warnings", "error",
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


def test_module_never_calls_ollama():
    source = Path(pypi_retriever.__file__).read_text(encoding="utf-8")

    assert "ollama" not in source.lower()
    assert "11434" not in source


# --- retrieve(): wrong_version compatibility-evidence intersection ----------

def _scipy_payload(versions, requires_python=">=3.10"):
    return {
        "files": [
            {"filename": f"scipy-{v}.tar.gz", "yanked": False, "requires-python": requires_python}
            for v in versions
        ]
    }


def _numpy_payload(versions, requires_python=">=3.10"):
    return {
        "files": [
            {"filename": f"numpy-{v}.tar.gz", "yanked": False, "requires-python": requires_python}
            for v in versions
        ]
    }


def test_wrong_version_cumtrapz_intersects_with_scipy_compatibility_evidence(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return FakeResponse(json.dumps(_scipy_payload(["1.13.0", "1.13.1", "1.14.0"])).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    result = retrieve(
        "scipy", python_version="3.10", subtype="wrong_version",
        module_path="scipy.integrate", symbol="cumtrapz",
    )

    assert result["status"] == "resolved"
    assert result["compatibility_evidence"]["status"] == "resolved"
    assert result["compatibility_evidence"]["compatible_specifier"] == "<1.14.0"
    versions = {c["version"] for c in result["candidate_versions"]}
    assert versions == {"1.13.0", "1.13.1"}
    assert "1.14.0" not in versions


def test_wrong_version_isshape_intersects_with_scipy_compatibility_evidence(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return FakeResponse(json.dumps(_scipy_payload(["1.13.1", "1.14.0", "1.14.1"])).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    result = retrieve(
        "scipy", python_version="3.10", subtype="wrong_version",
        module_path="scipy.sparse.sputils", symbol="isshape",
    )

    assert result["status"] == "resolved"
    versions = {c["version"] for c in result["candidate_versions"]}
    assert versions == {"1.13.1"}


def test_wrong_version_visible_deprecation_warning_intersects_with_numpy_compatibility_evidence(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return FakeResponse(json.dumps(_numpy_payload(["1.26.4", "2.0.0", "2.1.0"])).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    result = retrieve(
        "numpy", python_version="3.10", subtype="wrong_version",
        module_path="numpy", symbol="VisibleDeprecationWarning",
    )

    assert result["status"] == "resolved"
    assert result["compatibility_evidence"]["compatible_specifier"] == "<2.0.0"
    versions = {c["version"] for c in result["candidate_versions"]}
    assert versions == {"1.26.4"}
    assert "2.0.0" not in versions
    assert "2.1.0" not in versions


def test_wrong_version_intersection_applied_before_five_item_cap_not_after(monkeypatch):
    """The core ordering requirement: seven releases, the five newest are
    all API-incompatible (>=1.14.0). If the five-item cap were applied
    before the compatibility intersection, the two older compatible
    releases (1.13.0, 1.13.1) would already be gone and the result would be
    wrongly empty."""
    versions = ["1.13.0", "1.13.1", "1.14.0", "1.14.1", "1.15.0", "1.15.1", "1.16.0"]

    def fake_urlopen(request, timeout=None):
        return FakeResponse(json.dumps(_scipy_payload(versions)).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    result = retrieve(
        "scipy", python_version="3.10", subtype="wrong_version",
        module_path="scipy.integrate", symbol="cumtrapz",
    )

    survivors = {c["version"] for c in result["candidate_versions"]}
    assert survivors == {"1.13.0", "1.13.1"}
    assert result["status"] == "resolved"


def test_wrong_version_missing_evidence_returns_safe_abstention(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return FakeResponse(json.dumps(_scipy_payload(["1.13.0"])).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    result = retrieve(
        "scipy", python_version="3.10", subtype="wrong_version",
        module_path="scipy.some.unregistered", symbol="not_a_real_symbol",
    )

    assert result["status"] == "no_compatible_release"
    assert result["candidate_versions"] == []
    assert result["compatibility_evidence"]["status"] == "no_evidence"
    assert result["error"]


def test_wrong_version_missing_module_path_or_symbol_returns_safe_abstention(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("must not need to query PyPI to know the request is malformed")

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fail_if_called)

    result = retrieve("scipy", python_version="3.10", subtype="wrong_version")

    assert result["status"] == "no_compatible_release"
    assert result["error"]


def test_wrong_version_empty_pypi_intersection_returns_no_compatible_release(monkeypatch):
    """Evidence is resolved, but every PyPI release for this distribution is
    newer than the compatibility boundary - intersection is empty."""
    def fake_urlopen(request, timeout=None):
        return FakeResponse(json.dumps(_scipy_payload(["1.15.0", "1.16.0"])).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    result = retrieve(
        "scipy", python_version="3.10", subtype="wrong_version",
        module_path="scipy.integrate", symbol="cumtrapz",
    )

    assert result["status"] == "no_compatible_release"
    assert result["candidate_versions"] == []
    assert result["compatibility_evidence"]["status"] == "resolved"


def test_wrong_version_never_returns_a_command_field(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return FakeResponse(json.dumps(_scipy_payload(["1.13.0"])).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    result = retrieve(
        "scipy", python_version="3.10", subtype="wrong_version",
        module_path="scipy.integrate", symbol="cumtrapz",
    )

    assert "command" not in result
    for candidate in result["candidate_versions"]:
        assert "command" not in candidate


# --- offline production-path coverage for the five L5 PoC names -------------
# Issue #42 checklist item 9 asks for the five L5 PoC names (sklearn, umap,
# pkg_resources, scipy, dms_variants) to be re-run against the production
# module. sklearn (see test_retrieve_resolved_path_returns_full_schema and
# the wrong_version tests above, which use scipy) and dms_variants (see
# test_retrieve_mapping_unknown_makes_no_network_request) already go through
# retrieve() elsewhere in this file. The two tests below close the remaining
# gap for umap and pkg_resources, which were previously only asserted in the
# mapping table, never run through retrieve() itself.
#
# These are mocked/offline confirmations only - they prove retrieve()'s own
# logic (mapping, endpoint, filtering, schema) behaves correctly for these
# names. They are not the literal live PoC re-run issue #42 also asks for;
# that comparison against the L5 PoC's own recorded findings (docs/rag-design.md
# §9-§12) is performed separately, against the real PyPI API, as part of the
# upcoming manual pilot - see the accompanying commit's final report.

def test_retrieve_umap_resolves_to_umap_learn_and_applies_filtering(monkeypatch):
    captured = {}
    payload = {
        "files": [
            {"filename": "umap_learn-0.5.9-py3-none-any.whl", "yanked": False, "requires-python": ">=3.10"},
            {"filename": "umap_learn-0.5.0-py3-none-any.whl", "yanked": True, "requires-python": ">=3.10"},
        ]
    }

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        return FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    result = retrieve("umap", python_version="3.10")

    assert result["distribution_name"] == "umap-learn"
    assert captured["url"] == "https://pypi.org/simple/umap-learn/"
    assert result["status"] == "resolved"
    versions = [c["version"] for c in result["candidate_versions"]]
    assert versions == ["0.5.9"]  # the yanked 0.5.0 release is filtered out
    assert set(result.keys()) == {
        "status", "import_name", "subtype", "module_path", "symbol",
        "distribution_name", "package_found", "python_version",
        "latest_version", "candidate_versions", "compatibility_evidence",
        "source_endpoint", "retrieved_at", "warnings", "error",
    }


def test_retrieve_pkg_resources_resolves_to_setuptools_and_applies_filtering(monkeypatch):
    captured = {}
    payload = {
        "files": [
            {"filename": "setuptools-80.0.0.tar.gz", "yanked": False, "requires-python": ">=3.9"},
            {"filename": "setuptools-79.0.0-py3-none-any.whl", "yanked": False, "requires-python": "<3.8"},
        ]
    }

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        return FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    result = retrieve("pkg_resources", python_version="3.10")

    assert result["distribution_name"] == "setuptools"
    assert captured["url"] == "https://pypi.org/simple/setuptools/"
    assert result["status"] == "resolved"
    versions = [c["version"] for c in result["candidate_versions"]]
    assert versions == ["80.0.0"]  # 79.0.0 is filtered out (requires-python <3.8 excludes 3.10)
    assert set(result.keys()) == {
        "status", "import_name", "subtype", "module_path", "symbol",
        "distribution_name", "package_found", "python_version",
        "latest_version", "candidate_versions", "compatibility_evidence",
        "source_endpoint", "retrieved_at", "warnings", "error",
    }


# --- PyPI client rate limiting (i4, issue #42 checklist item 10) ------------

def test_default_rate_limit_config_is_loaded_from_rag_repair_yaml():
    assert pypi_retriever.DEFAULT_MIN_REQUEST_INTERVAL_SECONDS == 0.5
    assert pypi_retriever.DEFAULT_MAX_RETRIES_ON_429 == 1
    assert pypi_retriever.DEFAULT_MAX_RETRY_AFTER_SECONDS == 5.0


def test_two_uncached_projects_are_throttled_by_the_configured_interval(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(pypi_retriever.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    def fake_urlopen(request, timeout=None):
        return FakeResponse(json.dumps({"files": []}).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    fetch_pypi_project("project-one", min_request_interval=1.0)
    fetch_pypi_project("project-two", min_request_interval=1.0)

    assert len(sleep_calls) == 1
    # Almost no real time elapses between two in-process calls, so the
    # computed remaining wait should be very close to the full interval.
    assert 0.9 <= sleep_calls[0] <= 1.0


def test_cache_hit_for_the_same_project_never_sleeps(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(pypi_retriever.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    def fake_urlopen(request, timeout=None):
        return FakeResponse(json.dumps({"files": []}).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    fetch_pypi_project("scikit-learn", min_request_interval=5.0)
    fetch_pypi_project("scikit-learn", min_request_interval=5.0)
    fetch_pypi_project("scikit-learn", min_request_interval=5.0)

    assert sleep_calls == []


def test_zero_interval_disables_throttling(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(pypi_retriever.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    def fake_urlopen(request, timeout=None):
        return FakeResponse(json.dumps({"files": []}).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    fetch_pypi_project("project-a", min_request_interval=0)
    fetch_pypi_project("project-b", min_request_interval=0)

    assert sleep_calls == []


def _http_429(url, retry_after=None):
    headers = {}
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(url, 429, "Too Many Requests", headers, None)


def test_http_429_is_recognized_explicitly_without_a_new_status(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise _http_429(request.full_url, retry_after="120")

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    status, data = fetch_pypi_project(
        "scikit-learn", min_request_interval=0, max_retries_on_429=0,
    )

    assert status == "network_error"  # existing status contract preserved
    assert data["reason"] == "rate_limited"
    assert data["retry_after_seconds"] == 120.0


def test_bounded_retry_honors_retry_after_and_then_succeeds(monkeypatch):
    calls = {"count": 0}
    sleep_calls = []
    monkeypatch.setattr(pypi_retriever.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    def fake_urlopen(request, timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise _http_429(request.full_url, retry_after="2")
        return FakeResponse(json.dumps({"files": []}).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    status, data = fetch_pypi_project(
        "scikit-learn", min_request_interval=0, max_retries_on_429=1, max_retry_after_seconds=5.0,
    )

    assert status == "ok"
    assert calls["count"] == 2
    assert sleep_calls == [2.0]


def test_malformed_retry_after_fails_safe_without_retrying(monkeypatch):
    calls = {"count": 0}

    def fake_urlopen(request, timeout=None):
        calls["count"] += 1
        raise _http_429(request.full_url, retry_after="not-a-number")

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    status, data = fetch_pypi_project(
        "scikit-learn", min_request_interval=0, max_retries_on_429=1,
    )

    assert status == "network_error"
    assert calls["count"] == 1  # no retry attempted with an unknown wait time
    assert data["reason"] == "rate_limited"
    assert data["retry_after_seconds"] is None


def test_negative_retry_after_fails_safe_without_retrying(monkeypatch):
    calls = {"count": 0}

    def fake_urlopen(request, timeout=None):
        calls["count"] += 1
        raise _http_429(request.full_url, retry_after="-5")

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    status, data = fetch_pypi_project(
        "scikit-learn", min_request_interval=0, max_retries_on_429=1,
    )

    assert status == "network_error"
    assert calls["count"] == 1
    assert data["retry_after_seconds"] is None


def test_retry_after_larger_than_configured_maximum_is_not_honored(monkeypatch):
    calls = {"count": 0}

    def fake_urlopen(request, timeout=None):
        calls["count"] += 1
        raise _http_429(request.full_url, retry_after="9999")

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    status, data = fetch_pypi_project(
        "scikit-learn", min_request_interval=0, max_retries_on_429=1, max_retry_after_seconds=5.0,
    )

    assert status == "network_error"
    assert calls["count"] == 1  # exceeds max_retry_after_seconds - never waited on
    assert data["retry_after_seconds"] == 9999.0


def test_no_unbounded_retry_on_repeated_429(monkeypatch):
    calls = {"count": 0}

    def fake_urlopen(request, timeout=None):
        calls["count"] += 1
        raise _http_429(request.full_url, retry_after="1")

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    status, data = fetch_pypi_project(
        "scikit-learn", min_request_interval=0, max_retries_on_429=1, max_retry_after_seconds=5.0,
    )

    assert status == "network_error"
    assert calls["count"] == 2  # exactly one initial attempt + one bounded retry
    assert data["retries_attempted"] == 1
    assert data["max_retries"] == 1


def test_retrieve_surfaces_rate_limit_info_in_existing_error_and_warnings_fields(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise _http_429(request.full_url, retry_after="30")

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    result = retrieve("sklearn")

    assert result["status"] == "network_error"  # existing status contract preserved
    assert "429" in result["error"]
    assert result["warnings"] == [result["error"]]
    # no new top-level field was added to the retrieval-result schema
    assert set(result.keys()) == {
        "status", "import_name", "subtype", "module_path", "symbol",
        "distribution_name", "package_found", "python_version",
        "latest_version", "candidate_versions", "compatibility_evidence",
        "source_endpoint", "retrieved_at", "warnings", "error",
    }


def test_ordinary_network_error_message_is_unchanged_by_rate_limit_handling(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    result = retrieve("sklearn")

    assert result["status"] == "network_error"
    assert result["error"] == "Could not reach PyPI (timeout, DNS, or connection failure)."
    assert result["warnings"] == []


def test_timeout_behavior_is_unchanged_by_rate_limit_handling(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise socket.timeout("timed out")

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    status, data = fetch_pypi_project("scikit-learn", min_request_interval=0)

    assert status == "network_error"
    assert data is None


def test_default_autouse_no_op_sleep_prevents_real_waiting(monkeypatch):
    """Confirms the autouse fixture's default time.sleep no-op (not a
    per-test override) is what protects a test that triggers throttling
    without thinking about it - e.g. two real, uncached requests under the
    real config default interval (0.5s). Without that fixture, this test
    would actually block for close to half a second."""
    import time as real_time

    def fake_urlopen(request, timeout=None):
        return FakeResponse(json.dumps({"files": []}).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    start = real_time.monotonic()
    fetch_pypi_project("project-x")  # min_request_interval not overridden -> config default
    fetch_pypi_project("project-y")
    elapsed = real_time.monotonic() - start

    assert elapsed < 0.1


# --- zero-network / zero-side-effect guarantees ------------------------------

def test_retrieve_makes_exactly_one_network_call_per_new_distribution(monkeypatch):
    calls = {"count": 0}

    def fake_urlopen(request, timeout=None):
        calls["count"] += 1
        return FakeResponse(json.dumps(_scipy_payload(["1.13.0"])).encode("utf-8"))

    monkeypatch.setattr(pypi_retriever.urllib.request, "urlopen", fake_urlopen)

    retrieve("scipy", python_version="3.10", subtype="wrong_version",
              module_path="scipy.integrate", symbol="cumtrapz")

    assert calls["count"] == 1


def test_no_real_network_module_is_used_for_transport():
    """Guards against a future edit accidentally adding requests/httpx or
    similar - the only HTTP surface must remain urllib, which every test in
    this file mocks."""
    source = Path(pypi_retriever.__file__).read_text(encoding="utf-8")

    assert "import requests" not in source
    assert "import httpx" not in source
