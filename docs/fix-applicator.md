# FixApplicator (i5)

## Purpose

FixApplicator (O3 in `docs/architecture-note.md` §5) takes one *already-validated, already-grounded*
repair proposal from RAGRepairAgent (i4) and answers exactly one question: **did applying this
proposed fix, inside an environment equivalent to the one that originally failed, actually make the
notebook run further?**

It does not decide *what* to install - that judgment (grounded in real PyPI/compatibility evidence)
was already made and validated by i4. FixApplicator's only job is: apply the fix, re-run the
notebook top to bottom, and classify what happened. No LLM call exists anywhere in this component -
it is a small, fully deterministic pipeline, matching the "procedural pipeline, not a self-directing
agent" principle in `docs/architecture-note.md` §5.

## Relationship between i4 and i5

```text
RAGRepairAgent (i4)                    FixApplicator (i5)
--------------------                   ------------------
resolves package/version from PyPI  →  never re-derives install_name/version;
                                        only re-validates and re-executes them
proposes action/install_name/       →  consumes the persisted result record
  version/argv, schema+grounding        (data/repair-proposals/*.jsonl)
  validated
                                     →  re-joins against the i2 dataset for
                                        repository/notebook identity
                                     →  rebuilds a Docker environment,
                                        applies the fix, re-runs the notebook
                                     →  classifies fixed / still_failing /
                                        apply_error
```

## The actual i4 input shape (not the architecture doc's illustration)

`docs/architecture-note.md` §6.1 shows an illustrative fix object with top-level `import_name` and
`pypi_evidence` fields. That is **not** what `scripts/rag_repair_agent.py::run_repair_agent()`
actually persists. FixApplicator consumes the real, implemented shape:

- `notebook_execution_id` (top-level; the only join key i4 preserves)
- `status` (`success` / `abstained` / `failed`) - only `"success"` is ever executed
- `final_action` (`install` / `pin_version` / `none`) - `"none"` is always skipped
- `final_install_name`, `final_version`
- `argv`, `command` (i4's own already-validated argv/display string)
- nested `input` block (`error_type`, `failing_module`, etc.) - used as the "original failure"
  baseline for outcome comparison
- `retrieval_result` (i4's PyPI evidence) - not consumed by i5 at all; it exists purely as i4's own
  audit trail

FixApplicator never trusts the persisted `argv`/`install_name`/`version` blindly: it re-validates
`install_name`/`version` with `repair_proposal_validator.is_safe_token()` (the exact function i4
itself uses) and then rebuilds the argv from scratch via `rag_repair_agent.build_argv()` - the same
function i4 used to build it in the first place. If the rebuilt argv does not match the persisted
one, the record is rejected as `apply_error` rather than executed. This is a second, independent
enforcement of i4's own safety invariant, not a redesign of it.

## Dataset re-join (`notebook_execution_id`)

i4's persisted `input` block does **not** carry `repository_id`, `notebook_id`, `notebook_name`, or
`repository_url` forward - confirmed by inspecting `scripts/rag_repair_agent.py::_base_result()`.
FixApplicator re-joins each i4 record against
`data/context-classification/dependency_error_contexts.jsonl` by `notebook_execution_id`
(`fix_applicator.load_i2_index()` / `resolve_attempt()`) to recover those fields. A
`notebook_execution_id` that cannot be found in the i2 dataset, or a matching row missing any of
these fields, is treated as `apply_error` (`failure_stage: "join"`) - never guessed or defaulted.

## Reconstructed Docker environment

The original per-repository Docker containers/images from the upstream FAIR Jupyter pipeline
(`~/era/computational-reproducibility-pmc-docker`) no longer exist on this machine - confirmed via
`docker images`/`docker ps -a` showing no artifact from that pipeline, and via that pipeline's own
`scripts/extract_error_contexts.py` docstring ("repositories cloned during the original Docker
pipeline run were never persisted to this machine"). **FixApplicator therefore rebuilds an
equivalent environment from the same recipe on every attempt, rather than attaching to a container
that no longer exists.**

The recipe is a fresh reimplementation (not a copy) of that pipeline's `lib/docker.sh` /
`lib/entrypoint.sh`, inspected read-only and never modified:

- Same base image (`python:3.10-slim`), same `jupyter`/`nbdime` install, same
  `requirements.txt`-then-`setup.py` baseline install loop (each package/setup.py installed
  independently; one failure is logged and skipped, never aborts the baseline).
- Same notebook execution command: `jupyter nbconvert --to notebook --execute --allow-errors
  <notebook> --output <name>_output.ipynb` - the exact mechanism the reference pipeline uses, reused
  rather than reinvented (no papermill, no manual `ExecutePreprocessor`).
- Only **one** target notebook is executed per attempt (the specific failing notebook from the i2
  join), not every notebook in the repository.

Two deliberate structural deviations from the reference pipeline, both load-bearing for i5's own
requirements:

1. **Build context is isolated from the repository clone.** The reference pipeline drops
   `Dockerfile`/`entrypoint.sh` into the cloned repo directory itself and builds with the whole repo
   as build context. `docker_runner.py` instead builds from a small, separate directory containing
   only the generated `Dockerfile`/`entrypoint.sh`; the cloned repository is bind-mounted at `/app`
   at *run* time instead (exactly as the reference pipeline also does at run time). This avoids
   sending an entire git history as Docker build context.
2. **The container runs as its default (root) user**, omitting the reference pipeline's `docker run
   --user <host-uid>:<host-gid>` flag. That flag exists there so the pipeline's own `pip install
   --user` calls (reused verbatim in the baseline loop here) land somewhere the host-mapped,
   non-root user can write. i4's own fix argv - e.g. `["python", "-m", "pip", "install",
   "scikit-learn"]` - has no `--user` flag of its own, and i5 must never rewrite an argv that already
   passed i4's grounding validation. Running the (single-use, disposable) container as root lets that
   unmodified argv install successfully without this component inventing a new argv shape.

## Commit pinning

`config/fix_applicator.yaml`'s `upstream_docker_pipeline.db_path` points at the same sibling
pipeline's sqlite DB, which records each repository's `commit` (and `requirements`/`setups` file
lists) - none of which the i2 dataset carries. When a commit is recorded, FixApplicator clones the
full repository and then explicitly `git checkout`s that commit (`docker_runner.checkout_commit()`)
before building anything - it never silently stays on the default branch when a recorded commit
exists. If the checkout fails (bad SHA, unreachable, garbage-collected), that is a structured
`apply_error` (`failure_stage: "checkout"`), never a silent fallback.

The reference pipeline's own `process_repo()` does **not** pin to this commit (`git clone --depth
1`, no checkout step) even though it records one - so FixApplicator, when a commit is available and
checkable, achieves *higher* fidelity than the pipeline that originally produced the dataset.

The DB lookup itself is optional at runtime: a missing or unreadable DB file (or a locked one - see
"Known limitations" below) makes `commit`/`requirements`/`setups` simply unavailable for that
attempt, not a hard failure. Commit pinning is a "prefer when available" requirement, not a
prerequisite for every attempt.

## Fix application safety

- `install_name`/`version` are re-validated with `is_safe_token()` before anything else happens.
- The argv actually executed is always rebuilt via `build_argv()`, never the raw persisted string.
- `docker_runner.render_fix_command()` re-checks every dynamic argv token against a second, narrower
  allowlist (`is_safe_shell_word()`) immediately before writing it into the generated bash
  entrypoint, since it must additionally tolerate `=` (for `name==version` pins) that
  `is_safe_token()` deliberately excludes - this is a second, distinct trust boundary, not a
  replacement for the first.
- Every external process (`git`, `docker`) is invoked as an argv list through an injectable runner;
  `shell=True`, `os.system`, `eval`, and `exec` are asserted absent from both `docker_runner.py` and
  `fix_applicator.py` by a static source-scan test.
- A failed fix installation aborts the container script immediately (`FIX_INSTALL_FAILED`, exit 1) -
  the notebook is never executed afterward, since running it without the fix in place would validate
  nothing.
- `final_action == "none"` and `status != "success"` records never reach any subprocess call at all -
  proven by tests using a runner that raises on any invocation.

## Outcome semantics

Exactly three values, matching the `repair_attempts.outcome` contract in `docs/architecture-note.md`
§6.2 - no fourth value is ever invented:

| Outcome | Meaning |
|---|---|
| `fixed` | The re-executed notebook's output contains no error-type cell output at all. |
| `still_failing` | The fix installed and the notebook re-ran, but an error cell remains. `new_error_type`/`new_error_message` are populated from the first error cell found (`scripts/notebook_outcome.py`). `same_as_original_error` is `true` only when the new error's type matches the original **and** the original failing module name still appears in the new message - a conservative, substring-based check that only ever under-claims sameness, never over-claims it. |
| `apply_error` | The repair could not be meaningfully tested at all: dataset join failure, invalid i4 input, clone failure, commit checkout failure, Docker build failure, fix-install failure, container timeout, notebook-not-found, or output-notebook missing/malformed. `failure_stage` names exactly which stage failed. |

`status != "success"` and `final_action == "none"` records are not attempts at all - they are
recorded with `status: "skipped"` and a `skip_reason`, with `outcome` left `null`. This is
deliberate: inventing a fourth *outcome* value (e.g. `"skipped"`) would violate the
architecture's 3-value contract; a separate `status`/`skip_reason` pair keeps the distinction without
doing that.

## Environment isolation

Every attempt gets a fresh, uniquely-named work directory, container, and (subject to Docker's own
layer caching) image (`docker_runner.make_attempt_names()`, one random suffix per call). **Docker
image-ID reuse across attempts is expected and is not evidence against isolation** - Docker's build
cache may legitimately reuse identical layers; the actual isolation guarantee is the fresh
clone/work-dir and fresh container per attempt, both proven fresh by the unique per-attempt naming,
and both always cleaned up (`docker_runner.cleanup()`) in a `finally` block covering success,
`apply_error`, and timeout alike.

## Known limitations

- **Fidelity is best-effort, not exact.** The original containers were never persisted, so "the same
  environment" means "rebuilt from the same recipe, with the same base image and the same recorded
  commit when available" - not a literal re-attach to a preserved artifact.
- **A `still_failing` result with a *different* error than the original is common and expected**, not
  a bug: both real pilot cases below hit an unrelated, pre-existing dependency gap further into the
  notebook after i4's targeted error was genuinely resolved. This is exactly the information a later
  second repair round (i7/e3, out of scope here) would need.
- **The upstream sibling DB can be locked** by another process holding it open (observed directly
  during this component's own pilot - see `docs/i5-live-validation.md`). FixApplicator degrades
  gracefully (falls back to no commit/no baseline setup metadata) rather than crashing, but this
  means commit pinning is not always available even when a commit is, in principle, recorded.
- **Three environment-specific bugs were found and fixed only by running a real container**, not by
  the mocked unit suite alone (see `docs/i5-live-validation.md` for the full account): a
  locale-dependent Unicode decode crash on captured subprocess output, a Windows text-mode newline
  translation that corrupted the generated entrypoint's shebang line, and a `shutil.rmtree`
  parameter interaction (`ignore_errors=True` silently disables `onexc`) that left partial work
  directories behind. All three are covered by regression tests now, but they are a reminder that
  this component's mocked tests can prove logical correctness, never environment correctness -
  the real pilot step is not optional.
- **No orchestrator, no two-round loop, no SQLite/RDF persistence.** FixApplicator returns a
  self-contained result per attempt; wiring multiple attempts into a batch pipeline, feeding a
  `still_failing` result back into RAGRepairAgent for a second round, and persisting to
  `repair_attempts`/the KG are all explicitly out of scope for i5 (i7, e3, and i6 respectively).
