#!/usr/bin/env python3

"""Docker/git execution wrapper for FixApplicator (i5).

Every external process is invoked as an argv list through an injectable
`runner` callable (default: subprocess.run with shell disabled) - this
module never builds a shell string, never shells out via the OS command
interpreter, and never evaluates dynamic code. Tests inject a fake runner
so the automated suite never needs Docker, git, or network access.

Reference design (read-only inspection, never modified, never imported at
runtime): the sibling FAIR Jupyter Docker pipeline at
`~/era/computational-reproducibility-pmc-docker` (lib/docker.sh,
lib/entrypoint.sh, lib/repo.sh). Its per-repo containers/images were never
persisted on this machine (confirmed empty `docker images`/`docker ps -a`
for any FAIR-Jupyter-named artifact), so this module rebuilds an
equivalent environment from the same recipe on every attempt rather than
attaching to a container that no longer exists. Two deliberate deviations
from that reference, both documented in docs/fix-applicator.md:

1. The build context here is a small, isolated directory containing only
   the generated Dockerfile/entrypoint.sh - never the cloned repository
   itself (avoids sending a whole git history as Docker build context).
   The repo is instead bind-mounted at /app at *run* time, exactly as the
   reference pipeline also does.
2. `docker run` here does not pass `--user <host-uid>:<host-gid>` the way
   the reference pipeline does. That flag exists there so pip's own
   `--user` installs (still reused verbatim in the requirements loop
   below) land in a location the host-mapped, non-root user can write.
   Dropping it means the container runs as its default (root) user, which
   is what allows RAGRepairAgent's (i4) own unmodified argv - e.g.
   ["python", "-m", "pip", "install", "scikit-learn"], with no --user flag
   of its own - to install successfully without this module rewriting a
   value that already passed i4's grounding validation. The container is
   single-use and disposed of after every attempt, so running it as root
   carries no meaningful additional risk.
"""

import os
import re
import shutil
import stat
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

Runner = Callable[..., subprocess.CompletedProcess]


def default_runner(argv: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """subprocess.run with shell disabled. Decodes captured output as UTF-8
    with replacement rather than relying on text=True's locale-dependent
    default codec - git/docker/pip/jupyter output routinely contains
    non-ASCII bytes (progress bars, accented package metadata) that crash
    the default decoder on a non-UTF-8 host locale (observed on Windows'
    default cp1252 during this component's own real-container pilot; see
    docs/i5-live-validation.md)."""
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    kwargs.setdefault("encoding", "utf-8")
    kwargs.setdefault("errors", "replace")
    return subprocess.run(argv, shell=False, **kwargs)


class DockerRunnerError(Exception):
    """Carries a `stage` (join/clone/checkout/build/run/timeout) and a
    human-readable `message` so callers can build a structured apply_error
    result without re-parsing exception text."""

    def __init__(self, stage: str, message: str):
        self.stage = stage
        self.message = message
        super().__init__(f"{stage}: {message}")


# --- naming -------------------------------------------------------------

# Port of the reference pipeline's sanitize_docker_name() (lib/repo.sh) to
# Python: lowercase, replace anything outside [a-z0-9._-] with '-', trim
# leading/trailing separators, collapse repeats, cap length.
_UNSAFE_DOCKER_CHARS_RE = re.compile(r"[^a-z0-9._-]")
_REPEATED_HYPHEN_RE = re.compile(r"-+")


def sanitize_docker_name(name: str) -> str:
    name = name.lower()
    name = _UNSAFE_DOCKER_CHARS_RE.sub("-", name)
    name = name.strip("-.")
    name = _REPEATED_HYPHEN_RE.sub("-", name)
    if not name:
        name = "unnamed"
    return name[:60]


def make_attempt_names(notebook_execution_id: Any) -> Dict[str, str]:
    """A fresh, attempt-specific name triple. The uuid suffix (not image-ID
    reuse) is what proves isolation between attempts - see the module
    docstring and docs/fix-applicator.md: Docker layer caching may
    legitimately reuse image layers/IDs across attempts, so image-ID
    uniqueness is never used as evidence of isolation here."""
    suffix = uuid.uuid4().hex[:8]
    base = sanitize_docker_name(f"i5-{notebook_execution_id}-{suffix}")
    return {
        "image_name": f"i5-image-{base}",
        "container_name": f"i5-container-{base}",
        "work_dir_name": f"i5-fixapply-{base}",
    }


# --- shell-embedding safety (defense in depth, narrower than is_safe_token) --

# repair_proposal_validator.is_safe_token() is the *primary* trust
# decision made by scripts/fix_applicator.py on install_name/version
# separately, before an argv is ever built. This is a second, narrower
# check made here, on the final argv actually about to be embedded as one
# bare word in a generated bash script line: it additionally allows '='
# (needed for pip's `name==version` syntax), which is_safe_token
# deliberately excludes, but still rejects anything containing shell
# metacharacters, quotes, or whitespace.
_SAFE_SHELL_WORD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._=-]*$")

_FIXED_ARGV_PREFIX = ("python", "-m", "pip", "install")


def is_safe_shell_word(token: str) -> bool:
    return bool(token) and bool(_SAFE_SHELL_WORD_RE.match(token))


def render_fix_command(fix_argv: List[str]) -> str:
    """Validate every element of an already-built pip argv (from
    scripts/rag_repair_agent.py::build_argv()) is safe to embed literally
    as bare words in a generated bash script line, then join them with
    spaces. Raises DockerRunnerError, refusing to write anything, if any
    element fails - this function never runs anything itself, it only
    prepares text for write_build_context()."""
    if not fix_argv or tuple(fix_argv[: len(_FIXED_ARGV_PREFIX)]) != _FIXED_ARGV_PREFIX:
        raise DockerRunnerError("build", f"fix argv does not match the expected pip-install shape: {fix_argv!r}")

    # Only the tokens after the fixed "python -m pip install" prefix are
    # ever attacker/data-influenced (install_name, or install_name==version);
    # the prefix itself is a hardcoded literal, not a candidate for the
    # is_safe_shell_word() charset (e.g. "-m" legitimately starts with '-').
    for token in fix_argv[len(_FIXED_ARGV_PREFIX) :]:
        if not is_safe_shell_word(token):
            raise DockerRunnerError("build", f"unsafe token in fix argv, refusing to build entrypoint: {token!r}")

    return " ".join(fix_argv)


# --- git -----------------------------------------------------------------

def clone_repository(
    repository_url: str,
    dest_dir: Path,
    runner: Runner = default_runner,
    timeout: float = 120.0,
) -> None:
    if not repository_url or not repository_url.startswith("https://github.com/"):
        raise DockerRunnerError("clone", f"unsupported or missing repository_url: {repository_url!r}")

    argv = ["git", "clone", repository_url, str(dest_dir)]
    try:
        result = runner(argv, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise DockerRunnerError("clone", f"git clone timed out after {timeout}s") from e

    if result.returncode != 0:
        raise DockerRunnerError("clone", f"git clone failed (exit {result.returncode}): {result.stderr}")


def checkout_commit(
    repo_dir: Path,
    commit: Optional[str],
    runner: Runner = default_runner,
    timeout: float = 60.0,
) -> str:
    """Returns "checked_out" or "skipped_no_commit". Raises
    DockerRunnerError on checkout failure - a recorded commit that cannot
    be checked out is never silently swapped for the default branch."""
    if not commit:
        return "skipped_no_commit"

    argv = ["git", "checkout", commit]
    try:
        result = runner(argv, cwd=str(repo_dir), timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise DockerRunnerError("checkout", f"git checkout {commit} timed out after {timeout}s") from e

    if result.returncode != 0:
        raise DockerRunnerError(
            "checkout", f"git checkout {commit} failed (exit {result.returncode}): {result.stderr}"
        )
    return "checked_out"


# --- Docker build context -------------------------------------------------

DOCKERFILE_TEMPLATE = """FROM python:3.10-slim

WORKDIR /app

ENV HOME=/tmp

RUN pip install --upgrade pip setuptools wheel --root-user-action=ignore
RUN pip install jupyter nbdime --root-user-action=ignore

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
"""

# Baseline requirements/setup.py install loop is a fresh re-implementation
# of the reference pipeline's lib/entrypoint.sh, not a copy of that file -
# same behavior (install what's declared, skip individual failures,
# continue), same jupyter nbconvert invocation, with exactly one inserted
# step (the FIX_COMMAND block) that the reference pipeline has no
# equivalent of. "set -e" only aborts the *outer* script on an unhandled
# nonzero exit; the baseline loop below deliberately wraps each install in
# its own if/else so one bad baseline package never aborts the attempt -
# the FIX_COMMAND block deliberately does NOT do this, since a failed
# repair install must stop the attempt immediately (never run the
# notebook afterward - see docs/fix-applicator.md).
ENTRYPOINT_TEMPLATE = """#!/bin/bash
set -e

echo "[ENTRYPOINT] Starting i5 fix-application run"

if [ -z "$NOTEBOOK_PATHS" ]; then
    echo "[ENTRYPOINT] No notebook path provided"
    exit 1
fi

export HOME=/tmp
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH"

echo "[ENTRYPOINT] Checking for requirements.txt"
if [ -f "/app/requirements.txt" ]; then
    echo "[ENTRYPOINT] Installing from requirements.txt (one by one)..."
    while IFS= read -r package || [ -n "$package" ]; do
        [[ -z "$package" ]] && continue
        [[ "$package" =~ ^[[:space:]]*# ]] && continue
        echo "[ENTRYPOINT] Installing: $package"
        if pip install --user --no-cache-dir "$package"; then
            echo "[ENTRYPOINT] Installed: $package"
        else
            echo "[ENTRYPOINT] Failed to install: $package (skipping)"
        fi
    done < /app/requirements.txt
else
    echo "[ENTRYPOINT] No requirements.txt found"
fi

if [ -n "$SETUP_PATHS" ]; then
    echo "[ENTRYPOINT] Processing setup.py files"
    IFS=';' read -ra SETUP_FILES <<< "$SETUP_PATHS"
    for setup_file in "${{SETUP_FILES[@]}}"; do
        setup_file=$(echo "$setup_file" | xargs)
        [ -z "$setup_file" ] && continue
        setup_dir="/app/$(dirname "$setup_file")"
        if [ -d "$setup_dir" ] && [ -f "$setup_dir/setup.py" ]; then
            echo "[ENTRYPOINT] Installing from $setup_dir"
            (cd "$setup_dir" && pip install --user --no-cache-dir .) || \\
                echo "[ENTRYPOINT] Failed to install from $setup_dir"
        else
            echo "[ENTRYPOINT] No setup.py found in $setup_dir"
        fi
    done
fi

echo "[ENTRYPOINT] Applying proposed repair fix"
echo "[FIX] Running: {fix_command}"
if {fix_command}; then
    echo "FIX_INSTALL_SUCCESS"
else
    echo "FIX_INSTALL_FAILED"
    exit 1
fi

NOTEBOOK="$NOTEBOOK_PATHS"
if [ ! -f "/app/$NOTEBOOK" ]; then
    echo "EXEC_FAIL|$NOTEBOOK|0|NOTEBOOK_NOT_FOUND"
    exit 1
fi

notebook_dir=$(dirname "/app/$NOTEBOOK")
base_name=$(basename "$NOTEBOOK" .ipynb)
output_filename="${{base_name}}_output.ipynb"
output_nb_path="$notebook_dir/$output_filename"

echo "[ENTRYPOINT] Executing $NOTEBOOK"

jupyter nbconvert \\
    --to notebook \\
    --execute \\
    --allow-errors \\
    "/app/$NOTEBOOK" \\
    --output "$output_filename"

if [ ! -f "$output_nb_path" ]; then
    echo "EXEC_FAIL|$NOTEBOOK|0|OUTPUT_NOT_CREATED"
    exit 1
fi

echo "[ENTRYPOINT] Completed notebook execution"
"""


def write_build_context(build_dir: Path, fix_argv: List[str]) -> None:
    """Write Dockerfile + entrypoint.sh into an isolated build_dir (never
    the cloned repository directory - see module docstring, deviation 1).
    Raises DockerRunnerError without writing anything if fix_argv is not
    safe to embed."""
    fix_command = render_fix_command(fix_argv)

    # newline="\n" is required here, not optional: Path.write_text()'s
    # default newline translation on Windows rewrites every "\n" to
    # "\r\n", which corrupts entrypoint.sh's "#!/bin/bash" shebang line
    # into "#!/bin/bash\r" - Docker then fails every container with `exec
    # /entrypoint.sh: no such file or directory` (it looks for an
    # interpreter literally named "/bin/bash\r"). Observed directly during
    # this component's own real-container pilot; see
    # docs/i5-live-validation.md.
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "Dockerfile").write_text(DOCKERFILE_TEMPLATE, encoding="utf-8", newline="\n")
    (build_dir / "entrypoint.sh").write_text(
        ENTRYPOINT_TEMPLATE.format(fix_command=fix_command), encoding="utf-8", newline="\n"
    )


def build_image(
    build_dir: Path,
    image_name: str,
    runner: Runner = default_runner,
    timeout: float = 600.0,
) -> None:
    argv = ["docker", "build", "-t", image_name, str(build_dir)]
    try:
        result = runner(argv, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise DockerRunnerError("build", f"docker build timed out after {timeout}s") from e

    if result.returncode != 0:
        raise DockerRunnerError("build", f"docker build failed (exit {result.returncode}): {result.stderr}")


class ContainerRunResult:
    def __init__(self, return_code: int, stdout: str, stderr: str, timed_out: bool):
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out


def run_container(
    image_name: str,
    container_name: str,
    repo_dir: Path,
    notebook_relpath: str,
    setup_paths: Optional[List[str]] = None,
    runner: Runner = default_runner,
    timeout: float = 900.0,
) -> ContainerRunResult:
    """Runs one disposable container. Deliberately omits `--user
    <host-uid>:<host-gid>` - see module docstring, deviation 2."""
    argv = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "-v",
        f"{repo_dir}:/app",
        "--env",
        f"NOTEBOOK_PATHS={notebook_relpath}",
    ]
    if setup_paths:
        argv += ["--env", f"SETUP_PATHS={';'.join(setup_paths)}"]
    argv.append(image_name)

    try:
        result = runner(argv, timeout=timeout)
        return ContainerRunResult(result.returncode, result.stdout or "", result.stderr or "", timed_out=False)
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        return ContainerRunResult(-1, stdout, stderr, timed_out=True)


def _clear_readonly_and_retry(func, path, exc):
    """shutil.rmtree onerror handler: a fresh `git clone` on Windows
    leaves some of .git's packed-object files read-only, which makes a
    plain rmtree silently leave them (and their parent directories)
    behind even with ignore_errors=True - observed directly during this
    component's own real-container pilot (see docs/i5-live-validation.md).
    Clearing the read-only bit and retrying the same operation is the
    standard fix; if that still fails, the exception is swallowed by the
    ignore_errors=True call site, matching this function's "best effort"
    contract."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def cleanup(
    container_name: Optional[str],
    work_dir: Optional[Path],
    runner: Runner = default_runner,
) -> None:
    """Best-effort cleanup - must never raise, and must always attempt
    both steps even if one fails. Called from a `finally` block for every
    attempt outcome: success, apply_error, and timeout alike."""
    if container_name:
        try:
            runner(["docker", "rm", "-f", container_name], timeout=30.0)
        except Exception:
            pass

    if work_dir:
        # Note: ignore_errors=True would suppress onexc entirely (it is
        # only ever invoked when ignore_errors is false) - the "best
        # effort, never raise" contract instead lives inside
        # _clear_readonly_and_retry() itself, which always swallows
        # whatever it cannot fix.
        shutil.rmtree(work_dir, ignore_errors=False, onexc=_clear_readonly_and_retry)
