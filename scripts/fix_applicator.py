#!/usr/bin/env python3

"""FixApplicator (i5): consumes one successful RAGRepairAgent (i4) repair
proposal, applies it inside a freshly-rebuilt Docker environment, re-runs
the target notebook top-to-bottom, and classifies the outcome.

Deterministic only - no LLM call anywhere in this module. RAGRepairAgent
(i4) already made every judgment call about *what* to install; this
module only decides *whether the already-validated fix, once applied,
made the original failure go away*. See docs/fix-applicator.md for the
full design and its relationship to i4/L3.
"""

import argparse
import json
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

import docker_runner
from docker_runner import DockerRunnerError
import notebook_outcome
from notebook_outcome import NotebookReadError
from rag_repair_agent import build_argv
from repair_proposal_validator import is_safe_token


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "fix_applicator.yaml"

SUPPORTED_ACTIONS = {"install", "pin_version"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_fix_applicator_config(path: Optional[str] = None) -> Dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def load_i2_index(path: str) -> Dict[int, Dict[str, Any]]:
    """Load the i2 dataset keyed by notebook_execution_id - the re-join
    scripts/rag_repair_agent.py's (i4) own persisted output does not carry
    enough fields to avoid (it drops repository_id/notebook_id/
    notebook_name/repository_url; see docs/fix-applicator.md)."""
    index: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            notebook_execution_id = row.get("notebook_execution_id")
            if notebook_execution_id is not None:
                index[int(notebook_execution_id)] = row
    return index


def default_repository_metadata_lookup(
    db_path: Optional[str],
) -> Callable[[int], Optional[Dict[str, Any]]]:
    """Build a lookup callable over the upstream sibling Docker pipeline's
    sqlite DB (config/fix_applicator.yaml, upstream_docker_pipeline.db_path).
    Connects lazily and only once; returns None for every repository id
    (never raises) if the DB is missing or unreadable, so a repair attempt
    proceeds without commit-pinning/baseline-setup metadata instead of
    failing outright - see the config file's own comment for why this is
    intentionally non-fatal."""
    resolved_path = Path(db_path).expanduser() if db_path else None
    cache: Dict[str, Any] = {}

    def _lookup(repository_id: int) -> Optional[Dict[str, Any]]:
        if "table" not in cache:
            table: Dict[int, Any] = {}
            if resolved_path is not None and resolved_path.is_file():
                try:
                    from extract_error_contexts import load_repository_metadata

                    connection = sqlite3.connect(f"file:{resolved_path}?mode=ro", uri=True)
                    try:
                        table = load_repository_metadata(connection)
                    finally:
                        connection.close()
                except Exception:
                    table = {}
            cache["table"] = table
        return cache["table"].get(repository_id)

    return _lookup


def _split_paths(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [p.strip() for p in raw.split(";") if p.strip()]


# --- stage 1: dataset join + input validation --------------------------------

def resolve_attempt(
    i4_record: Dict[str, Any],
    i2_index: Dict[int, Dict[str, Any]],
    repository_metadata_lookup: Optional[Callable[[int], Optional[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Decide, from one i4 result record plus the i2 dataset join, whether
    a Docker attempt should be made. Touches no subprocess, no Docker, no
    filesystem beyond the injected repository_metadata_lookup callable -
    fully unit-testable in isolation.

    Returns a dict with "decision" in {"execute", "skip", "apply_error"}:
    - "skip": status != "success" or final_action == "none" - this was
      never a repair attempt to begin with, not a failure of one.
    - "apply_error": a "success" record whose action/install_name/version/
      argv fail re-validation, or whose notebook_execution_id cannot be
      joined against the i2 dataset.
    - "execute": everything needed to run a Docker attempt is present and
      independently re-validated.
    """
    notebook_execution_id = i4_record.get("notebook_execution_id")
    status = i4_record.get("status")
    action = i4_record.get("final_action")

    if status != "success":
        return {
            "decision": "skip",
            "notebook_execution_id": notebook_execution_id,
            "skip_reason": "input_status_not_success",
        }

    if action == "none":
        return {
            "decision": "skip",
            "notebook_execution_id": notebook_execution_id,
            "skip_reason": "action_none",
        }

    if action not in SUPPORTED_ACTIONS:
        return _validation_error(notebook_execution_id, f"unsupported final_action: {action!r}")

    install_name = i4_record.get("final_install_name")
    version = i4_record.get("final_version")

    if not is_safe_token(install_name):
        return _validation_error(
            notebook_execution_id, f"final_install_name failed safety validation: {install_name!r}"
        )

    if action == "install" and version is not None:
        return _validation_error(notebook_execution_id, "action is 'install' but final_version is not null")

    if action == "pin_version" and not is_safe_token(version):
        return _validation_error(
            notebook_execution_id, f"final_version failed safety validation: {version!r}"
        )

    # Reconstruct argv independently, via i4's own build_argv(), from the
    # just-validated primitives - never trust the persisted "argv"/"command"
    # fields verbatim. A mismatch against the persisted argv is a strong
    # data-integrity signal (e.g. a hand-edited result file) and is
    # rejected rather than silently executed.
    rebuilt_argv = build_argv(action, install_name, version)
    persisted_argv = i4_record.get("argv")
    if persisted_argv is not None and persisted_argv != rebuilt_argv:
        return _validation_error(
            notebook_execution_id,
            f"persisted argv {persisted_argv!r} does not match the argv rebuilt "
            f"from validated fields {rebuilt_argv!r}",
        )

    if notebook_execution_id is None:
        return {
            "decision": "apply_error",
            "notebook_execution_id": None,
            "failure_stage": "join",
            "diagnostic_message": "i4 record has no notebook_execution_id",
        }

    dataset_row = i2_index.get(int(notebook_execution_id))
    if dataset_row is None:
        return {
            "decision": "apply_error",
            "notebook_execution_id": notebook_execution_id,
            "failure_stage": "join",
            "diagnostic_message": f"notebook_execution_id {notebook_execution_id} not found in i2 dataset",
        }

    repository_id = dataset_row.get("repository_id")
    notebook_name = dataset_row.get("notebook_name")
    repository_url = dataset_row.get("repository_url")

    if not repository_url or not notebook_name or repository_id is None:
        return {
            "decision": "apply_error",
            "notebook_execution_id": notebook_execution_id,
            "failure_stage": "join",
            "diagnostic_message": (
                "i2 dataset row is missing repository_id/notebook_name/repository_url "
                f"for notebook_execution_id {notebook_execution_id}"
            ),
        }

    repo_meta: Dict[str, Any] = {}
    if repository_metadata_lookup is not None:
        try:
            repo_meta = repository_metadata_lookup(repository_id) or {}
        except Exception:
            repo_meta = {}

    input_block = i4_record.get("input") or {}

    return {
        "decision": "execute",
        "notebook_execution_id": notebook_execution_id,
        "action": action,
        "install_name": install_name,
        "version": version,
        "argv": rebuilt_argv,
        "command": " ".join(rebuilt_argv),
        "original_error_type": input_block.get("error_type"),
        "original_failing_module": input_block.get("failing_module"),
        "repository_id": repository_id,
        "notebook_id": dataset_row.get("notebook_id"),
        "notebook_name": notebook_name,
        "repository_url": repository_url,
        "repository_commit": repo_meta.get("commit"),
        "requirements_paths": _split_paths(repo_meta.get("requirements")),
        "setup_paths": _split_paths(repo_meta.get("setups")),
    }


def _validation_error(notebook_execution_id: Any, message: str) -> Dict[str, Any]:
    return {
        "decision": "apply_error",
        "notebook_execution_id": notebook_execution_id,
        "failure_stage": "validation",
        "diagnostic_message": message,
    }


# --- result assembly ----------------------------------------------------------

def _base_attempt_result(notebook_execution_id: Any, run_id: str) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": utc_now(),
        "notebook_execution_id": notebook_execution_id,
        "status": None,
        "skip_reason": None,
        "outcome": None,
        "action": None,
        "install_name": None,
        "version": None,
        "argv": None,
        "command": None,
        "apply_return_code": None,
        "execution_status": None,
        "new_error_type": None,
        "new_error_message": None,
        "same_as_original_error": None,
        "repository_id": None,
        "notebook_id": None,
        "repository_url": None,
        "notebook_name": None,
        "repository_commit": None,
        "commit_checkout_status": None,
        "failure_stage": None,
        "diagnostic_message": None,
        "elapsed_seconds": None,
        "errors": [],
    }


def _output_notebook_path(repo_dir: Path, notebook_relpath: str) -> Path:
    notebook_path = Path(notebook_relpath)
    return repo_dir / notebook_path.parent / f"{notebook_path.stem}_output.ipynb"


def apply_and_validate(
    i4_record: Dict[str, Any],
    i2_index: Dict[int, Dict[str, Any]],
    config: Dict[str, Any],
    repository_metadata_lookup: Optional[Callable[[int], Optional[Dict[str, Any]]]] = None,
    runner: docker_runner.Runner = docker_runner.default_runner,
    run_id: Optional[str] = None,
    work_dir_base: Optional[Path] = None,
) -> Dict[str, Any]:
    """Full pipeline for one i4 record: resolve -> (clone -> checkout ->
    build -> run) -> classify -> assemble. Cleanup always runs, on every
    exit path, via the outer finally block."""
    run_id = run_id or "i5-{}".format(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    start = time.monotonic()

    attempt = resolve_attempt(i4_record, i2_index, repository_metadata_lookup)
    result = _base_attempt_result(attempt.get("notebook_execution_id"), run_id)

    if attempt["decision"] == "skip":
        result["status"] = "skipped"
        result["skip_reason"] = attempt["skip_reason"]
        return result

    if attempt["decision"] == "apply_error":
        result["status"] = "completed"
        result["outcome"] = "apply_error"
        result["failure_stage"] = attempt.get("failure_stage")
        result["diagnostic_message"] = attempt.get("diagnostic_message")
        result["errors"] = [attempt.get("diagnostic_message")]
        return result

    # decision == "execute"
    result.update(
        {
            "status": "completed",
            "action": attempt["action"],
            "install_name": attempt["install_name"],
            "version": attempt["version"],
            "argv": attempt["argv"],
            "command": attempt["command"],
            "repository_id": attempt["repository_id"],
            "notebook_id": attempt["notebook_id"],
            "repository_url": attempt["repository_url"],
            "notebook_name": attempt["notebook_name"],
            "repository_commit": attempt["repository_commit"],
        }
    )

    execution_cfg = config.get("execution", {})
    names = docker_runner.make_attempt_names(attempt["notebook_execution_id"])
    base_dir = Path(work_dir_base) if work_dir_base else Path(tempfile.gettempdir())
    work_dir = base_dir / names["work_dir_name"]
    repo_dir = work_dir / "repo"
    build_dir = work_dir / "build"
    container_name = names["container_name"]
    image_name = names["image_name"]

    try:
        try:
            docker_runner.clone_repository(
                attempt["repository_url"],
                repo_dir,
                runner=runner,
                timeout=execution_cfg.get("clone_timeout_seconds", 120),
            )

            result["commit_checkout_status"] = docker_runner.checkout_commit(
                repo_dir,
                attempt["repository_commit"],
                runner=runner,
                timeout=execution_cfg.get("checkout_timeout_seconds", 60),
            )

            docker_runner.write_build_context(build_dir, attempt["argv"])
            docker_runner.build_image(
                build_dir,
                image_name,
                runner=runner,
                timeout=execution_cfg.get("build_timeout_seconds", 600),
            )

            run_result = docker_runner.run_container(
                image_name,
                container_name,
                repo_dir,
                attempt["notebook_name"],
                setup_paths=attempt["setup_paths"],
                runner=runner,
                timeout=execution_cfg.get("run_timeout_seconds", 900),
            )
        except DockerRunnerError as e:
            result["outcome"] = "apply_error"
            result["failure_stage"] = e.stage
            result["diagnostic_message"] = e.message
            result["errors"] = [e.message]
            return result

        run_timeout = execution_cfg.get("run_timeout_seconds", 900)
        if run_result.timed_out:
            result["outcome"] = "apply_error"
            result["execution_status"] = "timeout"
            result["failure_stage"] = "timeout"
            result["diagnostic_message"] = f"container run exceeded {run_timeout}s"
            result["errors"] = [result["diagnostic_message"]]
            return result

        result["apply_return_code"] = run_result.return_code

        if "FIX_INSTALL_FAILED" in run_result.stdout:
            result["outcome"] = "apply_error"
            result["execution_status"] = "fix_install_failed"
            result["failure_stage"] = "fix_install"
            result["diagnostic_message"] = "the proposed fix could not be installed inside the container"
            result["errors"] = [result["diagnostic_message"]]
            return result

        if "NOTEBOOK_NOT_FOUND" in run_result.stdout:
            result["outcome"] = "apply_error"
            result["execution_status"] = "notebook_not_found"
            result["failure_stage"] = "notebook_execution"
            result["diagnostic_message"] = f"notebook not found in cloned repository: {attempt['notebook_name']}"
            result["errors"] = [result["diagnostic_message"]]
            return result

        if run_result.return_code != 0:
            result["outcome"] = "apply_error"
            result["execution_status"] = "docker_run_failed"
            result["failure_stage"] = "docker_run"
            result["diagnostic_message"] = f"container exited with code {run_result.return_code}"
            result["errors"] = [result["diagnostic_message"], (run_result.stdout or "")[-2000:]]
            return result

        output_notebook_path = _output_notebook_path(repo_dir, attempt["notebook_name"])
        try:
            notebook = notebook_outcome.load_output_notebook(output_notebook_path)
        except NotebookReadError as e:
            result["outcome"] = "apply_error"
            result["execution_status"] = "output_notebook_missing"
            result["failure_stage"] = "output_read"
            result["diagnostic_message"] = str(e)
            result["errors"] = [str(e)]
            return result

        classification = notebook_outcome.classify_outcome(
            notebook, attempt["original_error_type"], attempt["original_failing_module"]
        )
        result["execution_status"] = "success"
        result.update(classification)
        return result

    finally:
        docker_runner.cleanup(
            container_name, work_dir, runner=runner
        )
        result["elapsed_seconds"] = round(time.monotonic() - start, 3)


# --- CLI / batch runner -------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FixApplicator over RAGRepairAgent (i4) repair-proposal results."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--input", required=True, help="i4 result JSONL, e.g. data/repair-proposals/repair_proposals.jsonl"
    )
    parser.add_argument("--output")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_fix_applicator_config(args.config)
    i2_index = load_i2_index(
        config.get("dataset", {}).get("i2_path", "data/context-classification/dependency_error_contexts.jsonl")
    )
    repository_metadata_lookup = default_repository_metadata_lookup(
        config.get("upstream_docker_pipeline", {}).get("db_path")
    )

    output_path = Path(
        args.output or config.get("output", {}).get("path", "data/fix-attempts/fix_attempts.jsonl")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_id = "i5-{}".format(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    mode = "w" if args.overwrite else "a"
    processed = 0

    with output_path.open(mode, encoding="utf-8") as out, open(args.input, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index < args.start_index:
                continue
            if args.limit is not None and processed >= args.limit:
                break
            if not line.strip():
                continue

            i4_record = json.loads(line)
            result = apply_and_validate(
                i4_record, i2_index, config, repository_metadata_lookup, run_id=run_id
            )
            result["index"] = index

            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()

            print(
                "[{}] index={} status={} outcome={}".format(
                    utc_now(), index, result["status"], result.get("outcome")
                )
            )

            processed += 1


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print("ERROR: {}".format(e))
        raise SystemExit(1)
