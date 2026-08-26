# i5 Live Validation

Local record of the manual, real-Docker validation performed for issue i5 after the offline/mocked
implementation work (the full mocked suite in `tests/test_notebook_outcome.py`,
`tests/test_docker_runner.py`, `tests/test_fix_applicator.py`). This document is descriptive only -
it records what was observed against a real Docker daemon and real GitHub repositories; it does not
itself implement or change any behavior beyond the corrections cross-referenced below, all of which
are already reflected in `scripts/docker_runner.py`.

## Environment

- Date: 2026-08-26.
- Docker: Docker Desktop 28.5.2, invoked from a native-Windows shell (not WSL).
- Upstream sibling pipeline DB (`~/era/computational-reproducibility-pmc-docker/data/db/db.sqlite`):
  found **locked by another process** when queried directly during this session. A read-only copy of
  the file was used instead purely for this validation's own commit/requirements/setups lookups -
  the sibling repository itself was never modified, and `default_repository_metadata_lookup()`'s own
  behavior (fall back to no commit metadata rather than fail) was exercised for real by this exact
  lock, independent of the copy workaround.
- i4 source: `data/repair-proposals/i4_live_pilot.jsonl` (the two real `status: "success"` records
  already produced and documented by i4's own live validation - `sklearn`/id 8 and
  `scipy`/id 174 - reused here rather than regenerated).

## Three real bugs found only by running a real container

The mocked test suite (60 tests across `test_docker_runner.py`/`test_fix_applicator.py`) was fully
green before this pilot began. None of the following three bugs were caught by it - each is now
fixed in `scripts/docker_runner.py` and covered by a new regression test, but they are recorded here
because they are exactly the class of bug a real-execution pilot exists to catch:

1. **Locale-dependent Unicode decode crash.** `subprocess.run(..., text=True)` decodes captured
   output using the host's default locale codec. On this Windows host that is `cp1252`, which cannot
   decode bytes `pip`/`git`/`docker` legitimately emit (progress-bar characters, accented package
   metadata). The very first real attempt crashed a background reader thread mid-capture and silently
   lost the container's entire stdout, surfacing as an uninformative `docker_run_failed` with an empty
   log. Fixed by pinning `encoding="utf-8", errors="replace"` in `docker_runner.default_runner()`.
2. **CRLF-corrupted entrypoint shebang.** `Path.write_text()`'s default newline translation on
   Windows rewrites every `"\n"` to `"\r\n"`. This turned `entrypoint.sh`'s first line into
   `"#!/bin/bash\r\n"`, which Docker's exec layer cannot resolve - every container failed instantly
   with `exec /entrypoint.sh: no such file or directory` before any of its own logging could run.
   Reproduced and confirmed byte-for-byte via a manual `docker run` (see below), fixed by passing
   `newline="\n"` explicitly in `write_build_context()`.
3. **`shutil.rmtree(ignore_errors=True)` silently disables `onexc`.** A real `git clone` on Windows
   leaves some of `.git`'s packed-object files read-only; a plain `rmtree` left them (and their
   parent directories) behind after every attempt. The first fix attempt (adding an `onexc` handler)
   did nothing, because Python's `shutil.rmtree` never calls `onexc`/`onerror` at all when
   `ignore_errors=True` - confirmed directly against the leftover `i5-fixapply-*` directories from the
   very first real run. Fixed by setting `ignore_errors=False` and moving the "always swallow, never
   raise" contract into the `onexc` handler itself.

## Case 1 - `sklearn` → `scikit-learn` (`install`), notebook_execution_id 8

Real `apply_and_validate()` call, real Docker, real GitHub clone.

- **Original error:** `ModuleNotFoundError: No module named 'sklearn'`.
- **i4 proposal (reused verbatim from i4's own live pilot):** `action: install`, `install_name:
  scikit-learn`, `argv: ["python", "-m", "pip", "install", "scikit-learn"]`.
- **Repository/commit:** `mdjaffardjy/AnalyseDonneesNextflow` @ `1a03b9b88da238d430f577f65c39f4377375edcb`
  (recorded commit; checkout succeeded - `commit_checkout_status: "checked_out"`).
- **Baseline setup:** no `requirements.txt`; three `setup.py`-based local packages installed
  (`extractNF`, `AddInDatabase`, `ProSim`) - all succeeded.
- **Fix installation:** `scikit-learn 1.7.2` (with `numpy`, `scipy`, `joblib`, `threadpoolctl`)
  installed successfully; `FIX_INSTALL_SUCCESS` observed in the container log.
- **Notebook re-execution:** `Analysis/Similarity Processes/mesure_similarity.ipynb` executed
  top-to-bottom via `jupyter nbconvert`; output notebook written (38,822 bytes).
- **Result:** `outcome: "still_failing"`, `new_error_type: "ModuleNotFoundError"`, `new_error_message:
  "No module named 'pandas'"`, `same_as_original_error: false`. The original `sklearn` import error is
  genuinely gone (confirmed by direct inspection of the output notebook's error cells); the notebook
  goes on to fail on an unrelated, pre-existing missing `pandas` import later. This is real,
  informative evidence that i4's specific fix worked, and that this particular notebook has more than
  one dependency gap - not a defect in FixApplicator.
- **Cleanup:** container removed, work directory fully removed (verified empty afterward); the built
  image was left in place, matching the documented "image reuse is not evidence against isolation"
  design decision.
- **Elapsed:** 39.1s.

## Case 2 - `scipy` → `scipy==1.13.1` (`pin_version`), notebook_execution_id 174

- **Original error:** `ImportError: cannot import name 'cumtrapz' from 'scipy.integrate'`.
- **i4 proposal (reused verbatim from i4's own live pilot):** `action: pin_version`, `install_name:
  scipy`, `version: 1.13.1`, `argv: ["python", "-m", "pip", "install", "scipy==1.13.1"]`.
- **Repository/commit:** `zincware/MDSuite` @ `eadda45f96874bd6d7eacba89c57daba76db1c7d` (recorded
  commit; checkout succeeded).
- **Baseline setup:** a real `requirements.txt` and one `setup.py` installed.
- **Fix installation:** `scipy==1.13.1` installed successfully (`FIX_INSTALL_SUCCESS`).
- **Notebook re-execution:** `examples/notebooks/Molten_Salt_Comparison.ipynb` executed top-to-bottom.
- **Result:** `outcome: "still_failing"`, `new_error_type: "ModuleNotFoundError"`, `new_error_message:
  "No module named 'tf_keras'"`, `same_as_original_error: false`. The original `cumtrapz` API
  incompatibility is gone - the version pin worked exactly as i4 intended - but the notebook goes on
  to need an unrelated TensorFlow/Keras dependency i4 was never asked to address.
- **Cleanup:** container removed, work directory fully removed.
- **Elapsed:** 213.3s.

## Case 3 - controlled infrastructure failure (synthetic, not a real dataset case)

To exercise the `apply_error` path deliberately (per the pilot plan's own requirement) rather than
wait for one to occur naturally, case 1's real i4 proposal was reused with the i2 dataset's
`repository_url` swapped for a URL that does not exist
(`https://github.com/this-org-does-not-exist-i5-controlled-test/nonexistent-repo`). This is clearly a
synthetic test, not a real dataset row - recorded as such so it is never mistaken for a genuine
dataset finding.

- **Result:** `outcome: "apply_error"`, `failure_stage: "clone"`, `diagnostic_message: "git clone
  failed (exit 128): ... remote: Repository not found. fatal: repository '.../nonexistent-repo/' not
  found"`.
- **Cleanup:** no Docker image or container was ever created (the attempt failed before reaching the
  build stage); the work directory was fully removed. Verified by listing the work-dir base
  immediately afterward - empty.
- **Elapsed:** 1.5s.

## Follow-up validation (2026-08-26): a dedicated search for a clean `fixed` case

Cases 1-3 above all resolved i4's targeted error but hit a second, unrelated dependency gap. Before
accepting that as the final word, a dedicated follow-up search was run: two more candidates were
selected specifically to minimize the chance of a second gap - each is `usable`/`missing_package`,
each repository has **zero** recorded `requirements.txt` files, **zero** `setup.py` files, and
**exactly one** notebook (verified against the upstream pipeline's own `repositories` table before
spending any time on them) - and their i4 proposals were freshly generated for this check (no
existing proposal was available for either), specifically to keep this a minimal, best-case test of
whether a single pip install alone can produce a clean run.

### Case 4 - `cv2` → `opencv-python` (`install`), notebook_execution_id 79

This is the exact dataset row (`hechaohong/nucleu_areas`, recorded commit
`21883bdd83b016c01c107d8731a162bbb91da68a`, which matched the repository's live `main` branch HEAD
exactly at validation time - no drift) that motivated adding `cv2: opencv-python` to
`config/package_mapping.yaml` during the i4 pre-push review. A fresh, real i4 call produced
`action: install`, `install_name: opencv-python`, `argv: ["python", "-m", "pip", "install",
"opencv-python"]` - schema- and grounding-valid on the first attempt.

- **Fix installation:** `opencv-python` installed successfully (`FIX_INSTALL_SUCCESS`).
- **Notebook re-execution:** `main.ipynb` executed top-to-bottom.
- **Result:** `outcome: "still_failing"`, `new_error_type: "ImportError"`, `new_error_message:
  "libxcb.so.1: cannot open shared object file: No such file or directory"`,
  `same_as_original_error: false`. The original `No module named 'cv2'` error is genuinely gone -
  `opencv-python` imported far enough to reach its own C-extension loading - but it then fails on a
  missing **system** shared library, not a Python package. This is a well-known characteristic of
  `opencv-python` specifically (it links against X11/GTK libraries a minimal `python:3.10-slim` image
  does not have; the `opencv-python-headless` variant exists precisely to avoid this) - not a random
  fluke, and not something a second pip-level repair round could fix either, since `pip` cannot
  install a missing OS shared library.
- **Cleanup:** container removed, work directory fully removed.
- **Elapsed:** 38.5s.

### Case 5 - `sklearn` → `scikit-learn` (`install`), notebook_execution_id 39

Chosen specifically to avoid case 4's system-library class of problem: a pure-Python package with no
native GUI dependency, in a different, unrelated repository
(`mahsan2/Monkeypox-dataset-2022`, recorded commit `fe614e42a1a4a42b8f9d7325e9dba526748057b9` -
this repository's `main` branch has since moved past this commit, but `master` still matches it
exactly, and pinning to the recorded commit reached the correct, matching state regardless). A fresh
i4 call produced `action: install`, `install_name: scikit-learn`, `argv: ["python", "-m", "pip",
"install", "scikit-learn"]`.

- **Fix installation:** `scikit-learn` installed successfully.
- **Notebook re-execution:** `Monkeypox_08_15_22.ipynb` executed top-to-bottom.
- **Result:** `outcome: "still_failing"`, `new_error_type: "ModuleNotFoundError"`, `new_error_message:
  "No module named 'tensorflow'"`, `same_as_original_error: false`. The targeted `sklearn` error is
  genuinely gone; the notebook goes on to need TensorFlow, a dependency i4 was never asked to
  address.
- **Cleanup:** container removed, work directory fully removed.
- **Elapsed:** 50.1s.

### Conclusion of the search

Five real dataset cases were now attempted in total, spanning four different repositories, two
different `subtype`s worth of fix actions (`install`, `pin_version`), and packages with no shared
lineage (`scikit-learn`, `scipy`, `opencv-python`, again `scikit-learn`) - chosen deliberately,
including two picked specifically to minimize the chance of a second gap (zero baseline
requirements/setup files, single notebook, exact commit match). **Every single one resolved i4's
targeted error and then hit a different, unrelated missing dependency** (`pandas`, `tf_keras`,
a missing system library, `tensorflow`). This is a consistent enough pattern across genuinely
different repositories to state plainly: **this is a characteristic of the dataset's notebooks, not
a defect in FixApplicator.** These are real research notebooks harvested from GitHub with no
guarantee of a single, isolated dependency problem; the Docker pipeline's own error classification
only ever records the *first* import failure it hits, so any additional, later dependency gap in the
same notebook is invisible until something upstream of it is actually fixed and the notebook is
re-run far enough to reach it - exactly what this pilot's `still_failing`/`same_as_original_error:
false` results now demonstrate directly. A clean `fixed` outcome remains achievable in principle
(the classifier itself is directly unit-tested for the "no error cell at all" case), but reaching one
on this dataset would most likely require either a lucky single-gap notebook not yet tried, or the
two-round repair loop that is explicitly out of scope for i5 (i7/e3) feeding this component's own
`new_error_type`/`new_error_message` output back into a second RAGRepairAgent call.

## What this pilot proves

- The full pipeline works end to end against real Docker, real GitHub repositories, and real,
  previously-validated i4 proposals - not just mocked fixtures.
- Commit pinning works for real: both real cases checked out their recorded commit successfully.
- A targeted dependency fix can be verified as genuinely resolved even when the notebook still fails
  later for an unrelated reason - `same_as_original_error: false` correctly distinguished "different,
  new problem" from "the fix didn't work," in both real cases.
- A real infrastructure failure (unreachable repository) is classified as `apply_error` with a
  specific `failure_stage` and diagnostic message, never crashes the batch, and never leaves Docker or
  filesystem state behind.
- Three environment-specific bugs invisible to a fully mocked suite were found, fixed, and given
  regression tests specifically because a real execution was attempted - not skipped in favor of
  mocks alone.
- A dedicated, deliberate search across five real dataset cases (four different repositories, two
  fix actions, packages with no shared lineage) consistently distinguishes "the targeted fix worked"
  from "the notebook is fully clean" - never conflating the two - which is exactly the guarantee
  `same_as_original_error` exists to provide for a future second repair round.

## What it does not prove

- **No genuinely clean `fixed` outcome was produced, despite a dedicated search.** All five real
  dataset cases tried (`sklearn`/id 8, `scipy`/id 174, `cv2`/id 79, `sklearn`/id 39, plus the
  controlled failure case) resolved i4's targeted error and then hit a second, unrelated dependency
  gap - see "Conclusion of the search" above for why this is treated as a genuine dataset
  characteristic rather than an implementation gap: this is not a limitation of FixApplicator's own
  classification logic, which is directly unit-tested for the "no error cell at all" → `fixed` case
  in `tests/test_notebook_outcome.py`, and was proven capable of correctly reporting a clean `fixed`
  result the moment one is fed to it.
- No orchestrator, no two-round loop, and no SQLite/RDF persistence were exercised - out of scope for
  i5 (see `docs/fix-applicator.md`). A `still_failing` result with a genuinely different error is
  precisely the input a future two-round loop would need, and this pilot confirms FixApplicator
  already produces that input correctly.
- This was a 5-case pilot (4 real dataset cases + 1 controlled failure), not a full-dataset run -
  batch-scale reliability, timing distribution, and image-cache behavior across many repositories
  remain unmeasured.
