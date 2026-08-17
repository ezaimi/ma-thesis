# Error Classifier & Context Extraction (i2)

## Purpose

This document describes the ErrorClassifier and context-extraction component built on top of
the i1 dependency-error dataset. It refines each row's subtype, attaches a root-cause hint and
confidence level, and attempts to recover source context (failing cell, nearby imports,
dependency files) for later use by the LLM explanation/repair prompts.

## Input Files

- `data/dependency-errors/dependency_errors.csv` (i1 output)
- `~/era/computational-reproducibility-pmc-docker/data/db/db.sqlite`, tables `executions` and `repositories`

## Output Files

- `data/context-classification/dependency_error_contexts.jsonl` - one enriched record per i1 row
- `data/context-classification/classification_summary.json` - counts and distributions
- `data/context-classification/manual_validation_sample.csv` - stratified sample for manual agreement checks

Every enriched record carries three fields through unchanged from the i1 CSV: `scope_status`
(`usable` / `excluded`), `exclusion_reason`, and `split` (`dev` / `evaluation` / `excluded`). These
are i1's repair-eligibility classification, not part of i2's own subtype refinement - i2 still
enriches and writes a record for every i1 row regardless of `scope_status`. The fields exist so
that a later `RAGRepairAgent` (i4) can decide whether to attempt repair without re-deriving
eligibility, while `LLMExplainer` (i3) continues to explain every row. See
`docs/architecture-note.md` §7.1 for the full explanation-vs-repair scope split.

## Classifier Rules

Subtypes reuse the i1 taxonomy: `missing_package`, `wrong_version`, `system_library`,
`mapping_unknown`, `out_of_scope`. Rules are applied in this order:

1. `.so` / "shared object file" in the message -> `system_library`
2. Failing module is a known ambiguous local name (`utils`, `statistics`, `config`, `helpers`, `src`) -> `mapping_unknown` (or kept as `out_of_scope` if i1 already excluded it for that reason)
3. "cannot import name" in the message -> `wrong_version`
4. `ModuleNotFoundError` / "no module named" -> `missing_package`
5. "install" / "required" phrasing (e.g. optional-dependency messages) -> `missing_package`
6. Any other `ImportError` -> `wrong_version` (treated as a transitive-dependency signal)
7. Anything else, or a missing failing module -> `out_of_scope` / `mapping_unknown` with `insufficient_context`

## Root-Cause Hints

- `direct_missing_package` - the failing module is genuinely absent and its import name matches its PyPI distribution name
- `import_distribution_name_mismatch` - the import name differs from the PyPI distribution name (e.g. `sklearn` -> `scikit-learn`, see `IMPORT_DISTRIBUTION_MISMATCHES` in the script)
- `version_or_api_incompatibility` - "cannot import name" style failures
- `transitive_dependency` - generic `ImportError` not matching the more specific patterns above
- `system_level_dependency` - missing shared object / system library
- `local_import_or_path_issue` - ambiguous local module names
- `insufficient_context` - not enough information in the error message to decide

## Confidence Logic

- `high` - direct pattern match on error type/message (missing-module, cannot-import-name, `.so` failures)
- `medium` - inferred from softer phrasing ("install", "required") or a generic `ImportError`
- `low` - ambiguous local modules or an empty/unusable failing module

## Context Extraction Tiers

1. **`legacy_execution_hint`** - look up the notebook in the legacy `executions` table (by `notebook_id`). A candidate row must first mention the `failing_module` string somewhere, then pass a stricter check that the row's *final reported error* actually matches the i1 row, not just its cell source: either (a) the exact i1 `error_message` appears in `executions.msg`, or (b) for `ModuleNotFoundError`, the exact pattern `No module named '<failing_module>'` (or double-quoted) appears, or (c) for "cannot import name" errors, the same `cannot import name '...' from '...'` phrase appears. A row that only mentions the module inside the cell's import statements - while the traceback's actual final error is for a different module - is rejected and counted in `rows_with_rejected_legacy_hint`. The embedded cell source (when accepted) is extracted from the "executing the following cell:" traceback block. Always low-confidence, never authoritative.
2. **`remote_fetched`** - only with `--fetch-remote`. Fetches the notebook JSON and dependency files from `raw.githubusercontent.com`, using the `repositories` table's `repository` (`owner/repo`) field, trying the stored `commit` first, then `main`, then `master`. Extracts the failing cell (by `error_cell_index`), nearby import cells, and a small window of surrounding cells.
3. **`metadata_only`** / **`source_not_found`** - `metadata_only` when no fetch was attempted at all; `source_not_found` when `--fetch-remote` was used but every fetch failed. Both still carry the full metadata already present in the i1 CSV (IDs, error type/message/cell index, failing module, subtype).

## Why `executions.msg` Is Only a Hint

`executions` is a legacy table with no timestamp column, inherited from an earlier/different
notebook-execution pass rather than this Docker pipeline's own re-execution. Cross-checking it
against the i1 dataset shows 162 of 214 notebooks (76%) have a matching `notebook_id`, but only
about 40% of those actually contain the expected `failing_module` string anywhere in the traceback
text - the rest show a different `reason` (`NameError`, `Exception`, `<Skipping notebook>`) or
reference a different module entirely.

Even a substring match on the module name is not enough: the module can appear in the *cell
source* (an import statement) while the traceback's actual final error is for a different module
in the same cell - e.g. a cell importing `pandas` before `sklearn` may fail on `pandas` while the
current i1 row records a `sklearn` failure for the same notebook. `find_legacy_hint` therefore
only accepts a row once its final reported error matches the i1 row (see the tier description
above); rejected candidates are counted separately in `rows_with_rejected_legacy_hint` so the gap
between "module mentioned" and "module actually failed" stays visible.

## Why Metadata-Only Context Is Sometimes Necessary

No actual notebook source is stored locally: the `notebooks`/`repositories` tables hold only
metadata (cell counts, hashes, semicolon-separated path lists), never file contents, and the
repositories cloned during the original Docker pipeline run were never persisted to this machine
(`data/repositories/` does not exist). Falling back to metadata-only context lets the pipeline
keep every row instead of dropping ones without recoverable source.

## Remote-Fetch Limitations

- The pipeline does not pin to the recorded `commit` when cloning; this script tries the commit
  first but falls back to `main`/`master`, so a fetched notebook may not exactly match the code
  that originally failed.
- No GitHub authentication is used, so fetching is subject to `raw.githubusercontent.com`'s
  anonymous rate limits if run over the full dataset repeatedly.
- `error_cell_index` indexes into the full `cells` array of the notebook JSON (all cell types,
  per the pipeline's own `enumerate(notebook.cells)` logic in `summary.py`), not a code-cell-only
  counter - this script indexes the same way, but a notebook edited since the original run can
  still shift cells and misalign the index.
- Off by default; only runs with `--fetch-remote`, and every fetch failure falls through silently
  to the next tier rather than raising.

## Known Edge Cases

- Repository renamed, deleted, or made private since the original execution
- Notebook edited after `error_cell_index` was recorded, shifting cell positions
- Empty or missing `failing_module` (falls back to `mapping_unknown` / `insufficient_context`)
- A `notebook_id` match in `executions` with no corresponding `failing_module` mention (ignored, tier falls through)
- Any unexpected per-row error (bad row data, DB error) falls back to a bare `metadata_only` record instead of failing the whole batch

The extraction script is `scripts/extract_error_contexts.py`.
