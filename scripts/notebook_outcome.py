#!/usr/bin/env python3

"""Pure notebook-output classification for FixApplicator (i5).

No subprocess, no Docker, no network call anywhere in this module - it only
reads an already-produced notebook JSON structure and decides whether the
dependency error that motivated the repair attempt is gone. Reading the
output .ipynb file itself is the only I/O this module performs; a missing
or malformed file raises NotebookReadError rather than being silently
treated as "fixed" or "still_failing" - scripts/fix_applicator.py maps that
exception to an apply_error result.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class NotebookReadError(Exception):
    """Raised when an output notebook cannot be read or is not a valid
    notebook structure."""


def load_output_notebook(path: Union[str, Path]) -> Dict[str, Any]:
    """Read and parse one output .ipynb file. Raises NotebookReadError for
    a missing file, unreadable file, invalid JSON, or JSON that is not a
    notebook (no 'cells' list) - never returns a partial/best-effort
    result for any of these."""
    path = Path(path)

    if not path.is_file():
        raise NotebookReadError(f"output notebook not found: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise NotebookReadError(f"could not read output notebook {path}: {e}") from e

    try:
        notebook = json.loads(text)
    except json.JSONDecodeError as e:
        raise NotebookReadError(f"output notebook {path} is not valid JSON: {e}") from e

    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise NotebookReadError(f"output notebook {path} is missing a 'cells' list")

    return notebook


def find_first_error_output(notebook: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Scan cells in notebook order; return {"ename", "evalue"} for the
    first cell whose outputs contain an output_type == "error" entry, or
    None if the notebook has no error output at all. Malformed individual
    cells/outputs are skipped rather than raised - only a malformed
    top-level notebook (see load_output_notebook) is treated as
    unclassifiable."""
    for cell in notebook.get("cells", []):
        if not isinstance(cell, dict):
            continue
        outputs = cell.get("outputs")
        if not isinstance(outputs, list):
            continue
        for output in outputs:
            if isinstance(output, dict) and output.get("output_type") == "error":
                return {
                    "ename": output.get("ename") or "",
                    "evalue": output.get("evalue") or "",
                }
    return None


def is_same_dependency_error(
    new_error_type: str,
    new_error_message: str,
    original_error_type: Optional[str],
    original_failing_module: Optional[str],
) -> bool:
    """True only if the new error is recognizably the same dependency
    failure the repair attempt targeted: the same exception type, and the
    original failing module name still present in the new message. A
    deliberately conservative, substring-based check - mirrors the
    never-guess evidence discipline in scripts/rag_repair_agent.py
    (CANNOT_IMPORT_NAME_RE). A false "different" is safer here than a
    false "same", since this feeds a possible second repair round rather
    than a final judgment."""
    if not original_error_type or new_error_type != original_error_type:
        return False
    if not original_failing_module:
        return False
    return original_failing_module in (new_error_message or "")


def classify_outcome(
    notebook: Dict[str, Any],
    original_error_type: Optional[str],
    original_failing_module: Optional[str],
) -> Dict[str, Any]:
    """Pure classification of an already-parsed output notebook into
    "fixed" or "still_failing". Never returns "apply_error" - a notebook
    that cannot be read/parsed at all is an infrastructure concern the
    caller decides about via NotebookReadError, not a classification this
    function makes."""
    error = find_first_error_output(notebook)

    if error is None:
        return {
            "outcome": "fixed",
            "new_error_type": None,
            "new_error_message": None,
            "same_as_original_error": False,
        }

    new_error_type = error["ename"]
    new_error_message = error["evalue"]
    same = is_same_dependency_error(
        new_error_type, new_error_message, original_error_type, original_failing_module
    )

    return {
        "outcome": "still_failing",
        "new_error_type": new_error_type,
        "new_error_message": new_error_message,
        "same_as_original_error": same,
    }
