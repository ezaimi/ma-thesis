import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import fix_applicator
from fix_applicator import apply_and_validate, load_fix_applicator_config, resolve_attempt


CONFIG = {"execution": {}}


def _completed(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def _i4_success(
    notebook_execution_id=8,
    action="install",
    install_name="scikit-learn",
    version=None,
    error_type="ModuleNotFoundError",
    failing_module="sklearn",
    argv=None,
):
    if argv is None:
        argv = (
            ["python", "-m", "pip", "install", install_name]
            if version is None
            else ["python", "-m", "pip", "install", f"{install_name}=={version}"]
        )
    return {
        "notebook_execution_id": notebook_execution_id,
        "status": "success",
        "final_action": action,
        "final_install_name": install_name,
        "final_version": version,
        "argv": argv,
        "command": " ".join(argv),
        "input": {"error_type": error_type, "failing_module": failing_module},
    }


def _i2_row(notebook_execution_id=8, repository_id=14, notebook_id=27, notebook_name="notebook.ipynb", repository_url="https://github.com/org/repo"):
    return {
        "notebook_execution_id": notebook_execution_id,
        "repository_id": repository_id,
        "notebook_id": notebook_id,
        "notebook_name": notebook_name,
        "repository_url": repository_url,
    }


def _clean_notebook_json():
    return json.dumps({"cells": [{"cell_type": "code", "outputs": []}], "nbformat": 4})


def _error_notebook_json(ename, evalue):
    return json.dumps(
        {
            "cells": [
                {
                    "cell_type": "code",
                    "outputs": [{"output_type": "error", "ename": ename, "evalue": evalue, "traceback": []}],
                }
            ],
            "nbformat": 4,
        }
    )


class RefusingRunner:
    """Raises on any call - used to prove a code path never reaches
    subprocess at all."""

    def __call__(self, argv, **kwargs):
        raise AssertionError(f"subprocess should never have been called, got: {argv}")


class ScriptedDockerRunner:
    """Dispatches on (argv[0], argv[1]) and, for a scripted 'docker run'
    call, writes a fake output notebook into the bind-mounted repo
    directory (parsed out of the '-v host:/app' argument) - emulating what
    a real container would produce on the host via the bind mount."""

    def __init__(self, run_container_writes=None, run_container_result=None, overrides=None):
        self.calls = []
        self.run_container_writes = run_container_writes  # (ename, evalue) or None for "fixed" or "no_file"
        self.run_container_result = run_container_result or _completed(returncode=0, stdout="FIX_INSTALL_SUCCESS")
        self.overrides = overrides or {}

    def __call__(self, argv, **kwargs):
        self.calls.append({"argv": argv, "kwargs": kwargs})
        key = (argv[0], argv[1] if len(argv) > 1 else None)

        if key in self.overrides:
            outcome = self.overrides[key]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        if key == ("git", "clone"):
            # A real `git clone` creates the destination directory; this
            # fake must too, since the later scripted "docker run" writes
            # the fake output notebook into that same directory (emulating
            # a real container's bind-mounted write).
            Path(argv[3]).mkdir(parents=True, exist_ok=True)
            return _completed(returncode=0)
        if key == ("git", "checkout"):
            return _completed(returncode=0)
        if key == ("docker", "build"):
            return _completed(returncode=0)
        if key == ("docker", "run"):
            host_repo_dir = self._extract_volume_host_path(argv)
            if self.run_container_writes == "fixed":
                (host_repo_dir / "notebook_output.ipynb").write_text(_clean_notebook_json(), encoding="utf-8")
            elif isinstance(self.run_container_writes, tuple):
                (host_repo_dir / "notebook_output.ipynb").write_text(
                    _error_notebook_json(*self.run_container_writes), encoding="utf-8"
                )
            elif self.run_container_writes == "malformed":
                (host_repo_dir / "notebook_output.ipynb").write_text("{not valid json", encoding="utf-8")
            # "no_file" (or None): write nothing
            return self.run_container_result
        if key == ("docker", "rm"):
            return _completed(returncode=0)

        raise AssertionError(f"unscripted call: {argv}")

    @staticmethod
    def _extract_volume_host_path(argv) -> Path:
        # Strip the known ":/app" suffix rather than splitting on the first
        # ":" - a Windows host path itself contains a drive-letter colon
        # (e.g. "C:\\Users\\...\\repo:/app").
        for i, item in enumerate(argv):
            if item == "-v":
                volume_arg = argv[i + 1]
                assert volume_arg.endswith(":/app")
                return Path(volume_arg[: -len(":/app")])
        raise AssertionError("no -v volume argument found")


# --- resolve_attempt: skip / apply_error paths (no subprocess involved) ------


def test_resolve_attempt_skips_when_status_not_success():
    attempt = resolve_attempt(_i4_success() | {"status": "abstained"}, {8: _i2_row()})
    assert attempt == {
        "decision": "skip",
        "notebook_execution_id": 8,
        "skip_reason": "input_status_not_success",
    }


def test_resolve_attempt_skips_when_action_none():
    record = _i4_success() | {"final_action": "none", "final_install_name": None, "argv": None}
    attempt = resolve_attempt(record, {8: _i2_row()})
    assert attempt["decision"] == "skip"
    assert attempt["skip_reason"] == "action_none"


def test_resolve_attempt_rejects_unsafe_install_name():
    record = _i4_success(install_name="scikit-learn; rm -rf /", argv=["python", "-m", "pip", "install", "scikit-learn; rm -rf /"])
    attempt = resolve_attempt(record, {8: _i2_row()})
    assert attempt["decision"] == "apply_error"
    assert attempt["failure_stage"] == "validation"


def test_resolve_attempt_rejects_unsafe_version():
    record = _i4_success(action="pin_version", install_name="scipy", version="1.13.1; rm -rf /", argv=["python", "-m", "pip", "install", "scipy==1.13.1; rm -rf /"])
    attempt = resolve_attempt(record, {8: _i2_row()})
    assert attempt["decision"] == "apply_error"
    assert attempt["failure_stage"] == "validation"


def test_resolve_attempt_rejects_argv_mismatch():
    record = _i4_success()
    record["argv"] = ["python", "-m", "pip", "install", "some-other-package"]
    attempt = resolve_attempt(record, {8: _i2_row()})
    assert attempt["decision"] == "apply_error"
    assert "does not match" in attempt["diagnostic_message"]


def test_resolve_attempt_join_failure_when_notebook_execution_id_missing_from_i2():
    attempt = resolve_attempt(_i4_success(notebook_execution_id=999), {8: _i2_row()})
    assert attempt["decision"] == "apply_error"
    assert attempt["failure_stage"] == "join"


def test_resolve_attempt_execute_rebuilds_argv_and_joins_repository_fields():
    attempt = resolve_attempt(_i4_success(), {8: _i2_row()})
    assert attempt["decision"] == "execute"
    assert attempt["argv"] == ["python", "-m", "pip", "install", "scikit-learn"]
    assert attempt["repository_id"] == 14
    assert attempt["notebook_name"] == "notebook.ipynb"
    assert attempt["repository_url"] == "https://github.com/org/repo"
    assert attempt["original_error_type"] == "ModuleNotFoundError"
    assert attempt["original_failing_module"] == "sklearn"


def test_resolve_attempt_uses_repository_metadata_lookup_for_commit():
    def lookup(repository_id):
        assert repository_id == 14
        return {"commit": "abc123", "requirements": "requirements.txt", "setups": "setup.py;sub/setup.py"}

    attempt = resolve_attempt(_i4_success(), {8: _i2_row()}, repository_metadata_lookup=lookup)
    assert attempt["repository_commit"] == "abc123"
    assert attempt["setup_paths"] == ["setup.py", "sub/setup.py"]


def test_resolve_attempt_lookup_failure_does_not_raise():
    def lookup(repository_id):
        raise RuntimeError("db unavailable")

    attempt = resolve_attempt(_i4_success(), {8: _i2_row()}, repository_metadata_lookup=lookup)
    assert attempt["decision"] == "execute"
    assert attempt["repository_commit"] is None


# --- apply_and_validate: gated paths never touch subprocess -----------------


def test_apply_and_validate_status_not_success_never_calls_subprocess():
    record = _i4_success() | {"status": "abstained"}
    result = apply_and_validate(record, {8: _i2_row()}, CONFIG, runner=RefusingRunner())
    assert result["status"] == "skipped"
    assert result["skip_reason"] == "input_status_not_success"
    assert result["outcome"] is None


def test_apply_and_validate_action_none_never_calls_subprocess():
    record = _i4_success() | {"final_action": "none", "final_install_name": None, "argv": None}
    result = apply_and_validate(record, {8: _i2_row()}, CONFIG, runner=RefusingRunner())
    assert result["status"] == "skipped"
    assert result["skip_reason"] == "action_none"


def test_apply_and_validate_unsafe_install_name_never_calls_subprocess():
    record = _i4_success(install_name="scikit-learn; rm -rf /", argv=["python", "-m", "pip", "install", "scikit-learn; rm -rf /"])
    result = apply_and_validate(record, {8: _i2_row()}, CONFIG, runner=RefusingRunner())
    assert result["outcome"] == "apply_error"
    assert result["failure_stage"] == "validation"


def test_apply_and_validate_join_failure_never_calls_subprocess():
    result = apply_and_validate(_i4_success(notebook_execution_id=999), {8: _i2_row()}, CONFIG, runner=RefusingRunner())
    assert result["outcome"] == "apply_error"
    assert result["failure_stage"] == "join"


# --- apply_and_validate: full Docker pipeline (scripted, no real Docker) ----


def test_apply_and_validate_fixed_outcome(tmp_path):
    runner = ScriptedDockerRunner(run_container_writes="fixed")
    result = apply_and_validate(_i4_success(), {8: _i2_row()}, CONFIG, runner=runner, work_dir_base=tmp_path)

    assert result["status"] == "completed"
    assert result["outcome"] == "fixed"
    assert result["execution_status"] == "success"
    assert result["apply_return_code"] == 0
    assert result["new_error_type"] is None
    # cleanup ran: docker rm was called and the work dir no longer exists
    assert any(c["argv"][:2] == ["docker", "rm"] for c in runner.calls)


def test_apply_and_validate_pin_version_argv_and_still_failing_same_error(tmp_path):
    record = _i4_success(action="pin_version", install_name="scipy", version="1.13.1")
    runner = ScriptedDockerRunner(run_container_writes=("ImportError", "cannot import name 'cumtrapz' from 'scipy.integrate'"))
    record["input"] = {"error_type": "ImportError", "failing_module": "cumtrapz"}

    result = apply_and_validate(record, {8: _i2_row()}, CONFIG, runner=runner, work_dir_base=tmp_path)

    assert result["argv"] == ["python", "-m", "pip", "install", "scipy==1.13.1"]
    assert result["outcome"] == "still_failing"
    assert result["same_as_original_error"] is True


def test_apply_and_validate_different_error_after_rerun(tmp_path):
    record = _i4_success()
    runner = ScriptedDockerRunner(run_container_writes=("NameError", "name 'foo' is not defined"))
    result = apply_and_validate(record, {8: _i2_row()}, CONFIG, runner=runner, work_dir_base=tmp_path)
    assert result["outcome"] == "still_failing"
    assert result["new_error_type"] == "NameError"
    assert result["same_as_original_error"] is False


def test_apply_and_validate_clone_failure(tmp_path):
    runner = ScriptedDockerRunner(overrides={("git", "clone"): _completed(returncode=128, stderr="fatal: not found")})
    result = apply_and_validate(_i4_success(), {8: _i2_row()}, CONFIG, runner=runner, work_dir_base=tmp_path)
    assert result["outcome"] == "apply_error"
    assert result["failure_stage"] == "clone"


def test_apply_and_validate_commit_checkout_failure(tmp_path):
    def lookup(repository_id):
        return {"commit": "deadbeef"}

    runner = ScriptedDockerRunner(overrides={("git", "checkout"): _completed(returncode=1, stderr="fatal: reference not a tree")})
    result = apply_and_validate(
        _i4_success(), {8: _i2_row()}, CONFIG, repository_metadata_lookup=lookup, runner=runner, work_dir_base=tmp_path
    )
    assert result["outcome"] == "apply_error"
    assert result["failure_stage"] == "checkout"


def test_apply_and_validate_docker_build_failure(tmp_path):
    runner = ScriptedDockerRunner(overrides={("docker", "build"): _completed(returncode=1, stderr="Dockerfile error")})
    result = apply_and_validate(_i4_success(), {8: _i2_row()}, CONFIG, runner=runner, work_dir_base=tmp_path)
    assert result["outcome"] == "apply_error"
    assert result["failure_stage"] == "build"


def test_apply_and_validate_fix_install_failure_stops_before_notebook(tmp_path):
    runner = ScriptedDockerRunner(
        run_container_writes="fixed",  # would prove "fixed" if reached - it must not be reached
        run_container_result=_completed(returncode=1, stdout="[FIX] Running: ...\nFIX_INSTALL_FAILED"),
    )
    result = apply_and_validate(_i4_success(), {8: _i2_row()}, CONFIG, runner=runner, work_dir_base=tmp_path)
    assert result["outcome"] == "apply_error"
    assert result["execution_status"] == "fix_install_failed"
    assert result["failure_stage"] == "fix_install"


def test_apply_and_validate_notebook_not_found(tmp_path):
    runner = ScriptedDockerRunner(
        run_container_writes="no_file",
        run_container_result=_completed(returncode=1, stdout="EXEC_FAIL|notebook.ipynb|0|NOTEBOOK_NOT_FOUND"),
    )
    result = apply_and_validate(_i4_success(), {8: _i2_row()}, CONFIG, runner=runner, work_dir_base=tmp_path)
    assert result["outcome"] == "apply_error"
    assert result["execution_status"] == "notebook_not_found"


def test_apply_and_validate_output_notebook_missing(tmp_path):
    runner = ScriptedDockerRunner(run_container_writes="no_file", run_container_result=_completed(returncode=0, stdout="FIX_INSTALL_SUCCESS"))
    result = apply_and_validate(_i4_success(), {8: _i2_row()}, CONFIG, runner=runner, work_dir_base=tmp_path)
    assert result["outcome"] == "apply_error"
    assert result["execution_status"] == "output_notebook_missing"


def test_apply_and_validate_malformed_output_notebook(tmp_path):
    runner = ScriptedDockerRunner(run_container_writes="malformed")
    result = apply_and_validate(_i4_success(), {8: _i2_row()}, CONFIG, runner=runner, work_dir_base=tmp_path)
    assert result["outcome"] == "apply_error"
    assert result["execution_status"] == "output_notebook_missing"


def test_apply_and_validate_generic_docker_run_failure(tmp_path):
    runner = ScriptedDockerRunner(
        run_container_writes="no_file",
        run_container_result=_completed(returncode=137, stdout="Killed"),
    )
    result = apply_and_validate(_i4_success(), {8: _i2_row()}, CONFIG, runner=runner, work_dir_base=tmp_path)
    assert result["outcome"] == "apply_error"
    assert result["execution_status"] == "docker_run_failed"


def test_apply_and_validate_timeout(tmp_path):
    runner = ScriptedDockerRunner(overrides={("docker", "run"): subprocess.TimeoutExpired(cmd="docker run", timeout=1)})
    result = apply_and_validate(_i4_success(), {8: _i2_row()}, CONFIG, runner=runner, work_dir_base=tmp_path)
    assert result["outcome"] == "apply_error"
    assert result["execution_status"] == "timeout"
    # cleanup still ran even on timeout
    assert any(c["argv"][:2] == ["docker", "rm"] for c in runner.calls)


def test_apply_and_validate_cleanup_runs_on_failure_removes_work_dir(tmp_path):
    runner = ScriptedDockerRunner(overrides={("docker", "build"): _completed(returncode=1, stderr="boom")})
    apply_and_validate(_i4_success(), {8: _i2_row()}, CONFIG, runner=runner, work_dir_base=tmp_path)
    # nothing left behind under the attempt's work dir base
    assert list(tmp_path.iterdir()) == []


def test_apply_and_validate_cleanup_runs_on_success_removes_work_dir(tmp_path):
    runner = ScriptedDockerRunner(run_container_writes="fixed")
    apply_and_validate(_i4_success(), {8: _i2_row()}, CONFIG, runner=runner, work_dir_base=tmp_path)
    assert list(tmp_path.iterdir()) == []


# --- config loader --------------------------------------------------------


def test_load_fix_applicator_config_reads_execution_section():
    config = load_fix_applicator_config()
    assert "execution" in config
    assert config["execution"]["run_timeout_seconds"] > 0
    assert config["output"]["path"]


# --- static safety scan --------------------------------------------------------


def test_module_never_uses_shell_true_or_dangerous_execution():
    source = Path(fix_applicator.__file__).read_text(encoding="utf-8")
    assert "shell=True" not in source
    assert "os.system(" not in source
    assert "eval(" not in source
    assert "exec(" not in source
