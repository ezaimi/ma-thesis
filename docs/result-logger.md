# ResultLogger & KG Enrichment (i6)

## 1. Purpose

ResultLogger collects the outputs already produced by the earlier thesis components -
ErrorClassifier (i2), LLMExplainer (i3), RAGRepairAgent (i4), and FixApplicator (i5) - and stores
one row per repair attempt in a SQLite `repair_attempts` table. It does not classify errors,
generate explanations, propose fixes, or apply anything itself; it only joins and persists what
those components already produced, so the result is queryable for evaluation and preserved as
provenance. A second stage (`scripts/export_repair_attempts_csv.py` +
`mapping/rml_mapping/repair_attempts.rml.ttl`) re-expresses those rows as RDF and links them into
the existing FAIR Jupyter Knowledge Graph.

## 2. Inputs

ResultLogger (`scripts/result_logger.py`) reads three or four JSONL files, each already produced by
an earlier component, and joins them with a different strategy per pair - chosen because the real
data does not allow a single uniform join key:

- **i2** (`data/context-classification/dependency_error_contexts.jsonl`, via
  `fix_applicator.load_i2_index()`) - the required base context. Every row in `repair_attempts` must
  resolve a `notebook_execution_id` against this dataset; a miss is a hard `ValueError`, never a
  guess.
- **i3** (`load_i3_index()`) - joined by `notebook_execution_id`. Safe because i2/i3 have exactly one
  row per notebook. `load_i3_index()` raises `AmbiguousExplanationError` if a given i3 file ever
  contains two explanation records for the same `notebook_execution_id`, rather than silently
  picking one.
- **i4** (`load_i4_records()`) - the **driving source**. ResultLogger produces exactly one
  `repair_attempts` row per line of the given i4 file, in file order, including abstained and failed
  attempts (i4 processes every i2 row, not only repair-eligible ones).
- **i5** (`load_i5_index_by_position()`, optional) - joined by **position**, not by
  `notebook_execution_id`. `scripts/fix_applicator.py` stamps each of its output records with the
  0-based line index of the i4 file it processed; ResultLogger uses that same index to find the
  matching i5 outcome for a given i4 line.

The positional i4/i5 join exists specifically because `notebook_execution_id` can legitimately
repeat across i4 lines - the real `data/repair-proposals/i4_live_pilot.jsonl` fixture contains two
separate attempts for `notebook_execution_id=8` (a timed-out run and a later successful re-run), each
with its own `run_id`. Joining by id alone would be ambiguous; joining by line position is not.
`log_repair_attempts()` produces one row per i4 line regardless, so this duplicate case yields two
distinct `repair_attempts` rows rather than being collapsed or cross-matched.

## 3. `repair_attempts` table

The full column list and SQL `CREATE TABLE` statement are defined in `docs/architecture-note.md`
section 6.2; this document does not repeat them. In implementation terms the columns fall into six
groups:

- **error classification** (from i2): `notebook_execution_id`, `failing_module`, `subtype`
- **explanation** (from i3): `explanation`
- **proposed repair** (from i4, or i5 when available): `action`, `install_name`, `version`,
  `command`, `rationale`
- **PyPI evidence** (from i4): `pypi_evidence`
- **repair outcome** (from i5, when available): `outcome`, `new_error_type`, `new_error_message`
- **model/prompt/run metadata**: `llm_model`, `prompt_strategy` (from i3), `round` (currently always
  `1`), `run_id`, `created_at` (see section 5)

`scripts/result_logger.py::create_table()` issues `CREATE TABLE IF NOT EXISTS`, matching the
architecture note's schema exactly; `REPAIR_ATTEMPT_COLUMNS` in that module is the single source of
truth for the column list and is reused by the CSV exporter (section 6).

## 4. JSON preservation

Two columns intentionally hold serialized JSON rather than a reduced/flattened value, so that
downstream evaluation and provenance work never has to go back to the original i3/i4 JSONL files for
detail that was available at logging time:

- **`explanation`** - the complete structured LLM explanation object (`summary`, `root_cause`,
  `evidence`, `failing_module`, `explanation_confidence`, `limitations`, per
  `schemas/explanation.schema.json`) is `json.dumps`-ed into this TEXT column, not reduced to a
  single field.
- **`pypi_evidence`** - the complete i4 `retrieval_result` object (distribution name, all candidate
  versions, compatibility evidence with its source citation, warnings, etc.) is serialized the same
  way, not the smaller three-field illustration shown in `docs/architecture-note.md` section 6.1.

Both are `NULL` when the corresponding upstream record does not exist or carries no value (e.g. an
abstained i4 record has no `retrieval_result`), never an empty string or an empty JSON object.

## 5. Run ID behavior

`repair_attempts.run_id` and `created_at` prefer the i5 record's own `run_id`/`created_at` when a
matching i5 outcome exists for that i4 line (`build_repair_attempt_row()`). When no i5 record exists
- an abstained or failed i4 attempt, or simply one not yet run through FixApplicator - both fields
fall back to the i4 record's own `run_id`/`created_at` instead of being left `NULL`. This keeps every
i4 attempt logged and traceable to the run that produced it, at the cost of `run_id` not always
meaning "the FixApplicator run" for a row with no `outcome`.

i2, i3, i4, and i5 remain unmodified: none of them accept a shared, externally supplied `run_id`
today, and each mints its own per CLI invocation. A shared, orchestration-level `run_id` threaded
through all stages of one pipeline pass is intentionally left for i7's orchestrator. See
`docs/architecture-note.md` section 7.3 for the full rationale.

## 6. SQLite to CSV export

`scripts/export_repair_attempts_csv.py` exports `repair_attempts` rows to a CSV file
(`export_repair_attempts_csv()`). CSV is the target format because every existing FAIR Jupyter KG
mapping (`mapping/rml_mapping/*.rml.ttl`, including `executions.rml.ttl`) reads from a CSV data
source (`rml:referenceFormulation ql:CSV`), not from a live database connection.

`repair_attempts` itself only stores `notebook_execution_id` (see section 3), but the KG identifies a
notebook by `notebook_id`. The exporter resolves `notebook_id` per row by re-joining
`notebook_execution_id` against the same i2 dataset ResultLogger itself requires
(`fix_applicator.load_i2_index()`), the same re-join pattern `FixApplicator` already uses for its own
metadata recovery. A `notebook_execution_id` missing from the given i2 file raises a `ValueError`
naming the offending `repair_attempts.id`, rather than exporting a row with no notebook link.

`CSV_COLUMNS` is `["id", "notebook_id"] + REPAIR_ATTEMPT_COLUMNS` - `notebook_execution_id` is
carried into the CSV for traceability even though the RML mapping does not reference it.

## 7. FAIR Jupyter KG enrichment

`mapping/rml_mapping/repair_attempts.rml.ttl` is modeled directly on the existing
`executions.rml.ttl` (same prefix block, same `rr:TriplesMap`/`rr:PredicateObjectMap` structure) and
defines two triples maps over the exported CSV:

- **The repair attempt itself** - subject `https://w3id.org/reproduceme/repairattempt_{id}`, typed
  both `repr:RepairAttempt` and `prov:Activity`.
- **The link from the notebook** - subject `https://w3id.org/reproduceme/notebook_{notebook_id}`
  (byte-for-byte the same IRI template `notebooks.rml.ttl` already uses, so the two coincide as the
  same node once loaded into the same graph), with one triple:
  `repr:hadRepairAttempt` -> `https://w3id.org/reproduceme/repairattempt_{id}`.

Predicate choices:

- `created_at` -> `prov:endedAtTime`, explicitly typed `xsd:dateTime` (the one place this mapping
  deviates from the untyped-literal style of the rest of the KG, since PROV-O's own spec gives
  `prov:endedAtTime` an `xsd:dateTime` range).
- `new_error_type` -> `repr:exception` and `new_error_message` -> `repr:msg`, reusing the same
  predicates `executions.rml.ttl` uses for a `CellExecution`'s own exception/message, rather than
  minting new ones for what is the same kind of fact observed after the fix.
- `llm_model` -> `repr:llmModel`, a plain literal - not a PROV-O Agent resource.
- New `repr:` predicates for every other repair-specific fact: `repr:failingModule`, `repr:subtype`,
  `repr:explanation`, `repr:fixAction`, `repr:installName`, `repr:fixVersion`, `repr:fixCommand`,
  `repr:fixRationale`, `repr:pypiEvidence`, `repr:fixOutcome`, `repr:promptStrategy`, `repr:round`,
  `repr:runId`.

## 8. Notebook ID alignment

`docs/architecture-note.md` section 9 left this as an open item: the FAIR Jupyter KG is generated
from the original conda-pipeline dataset, not the Docker pipeline this thesis builds on, so it was
not known whether the two share notebook identifiers. This was checked directly, not assumed: all
**214** rows of `data/context-classification/dependency_error_contexts.jsonl` were cross-checked
against the FAIR Jupyter KG's own `repositories.csv`/`notebooks.csv` by `repository_id` and
`notebook_id`, comparing the resolved repository path and notebook filename for an exact string
match. Result: **214/214 matched** on both `repository_id` -> `repository` and `notebook_id` ->
`name`, zero missing, zero mismatches. The Docker pipeline reuses the same `repositories`/`notebooks`
primary keys as the conda-based corpus the KG was built from. No crosswalk table is required to link
a repair attempt to its existing notebook node; `notebook_id` (recovered per section 6) is sufficient
on its own.

## 9. Validation

- Full Python test suite: **332 passed** (`pytest tests/`), including the pre-existing i1-i5 suites.
- `tests/test_result_logger.py` - loader behavior, the duplicate-`notebook_execution_id` regression
  case described in section 2, the i5-missing run_id fallback, and end-to-end
  `log_repair_attempts()` runs against temporary JSONL fixtures.
- `tests/test_export_repair_attempts_csv.py` - `notebook_id` resolution, CSV header/column order,
  `NULL` columns becoming empty CSV fields (not the string `"None"`), JSON blob round-tripping
  through the CSV writer/reader unchanged, and the missing-i2-context error.
- `morph_kgc` (the same RML engine `run_fairjupyter_kg.sh` uses) was installed and run against a real
  `repair_attempts.csv` built from the real i2/i3/i4 pilot data plus a labeled synthetic i5 fixture
  (no live i5 output exists yet - see `docs/architecture-note.md` section 7.2 on why). The generated
  RDF was inspected directly: `NULL` SQLite columns produce no triple at all for the corresponding
  predicate (not an empty-string literal), `repr:exception`/`repr:msg` appear only on the one row
  that actually has a post-repair error, and `prov:endedAtTime` is present and correctly typed
  `xsd:dateTime` for every row.
- Notebook links were verified against the **real, unmodified** `notebooks.rml.ttl`, run against the
  real `notebooks.csv` (627,127 triples), confirming the linked notebook IRIs exist there as
  `rdf:type repr:Notebook` with the expected `pav:retrievedFrom` repository link - not merely
  matching by string inspection.
- The FAIR Jupyter KG checkout was confirmed untouched (directory listing identical before/after;
  all validation output was written to a scratch location).

## 10. Scope boundary / i7

i6 does not implement:

- Orchestration of ErrorClassifier, LLMExplainer, RAGRepairAgent, and FixApplicator into a single
  pipeline pass.
- Automatic execution over the full dataset - ResultLogger and the CSV/RDF export are invoked against
  whatever i2/i3/i4/i5 JSONL files already exist, not run as part of a batch driver.
- The second repair attempt / bounded two-round repair loop (`round` is currently always `1`; no
  feedback from a `still_failing` outcome back into RAGRepairAgent exists).

These, along with the shared orchestration-level `run_id` noted in section 5, belong to i7.
