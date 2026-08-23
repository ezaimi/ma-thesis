# i4 Live Validation

Local record of the manual validation performed for issue #42 after the offline/mocked implementation
work in commits `0a38f06`..`18121d7`. This document is descriptive only - it records what was
observed during two manual checks against real external services; it does not itself implement or
change any behavior beyond the corrections cross-referenced below.

## Environment

- Date: 2026-08-17.
- Model: `gemma2:9b` via a local Ollama instance.
- Processor: CPU-only, observed at 100% utilization during generation.
- Runtime target: Python 3.10 (`config/rag_repair.yaml`'s `runtime.python_version`), matching the
  Docker pipeline's hardcoded interpreter - not the interpreter used to run this validation itself.
- The generated argv from every successful proposal below was inspected but **never executed** - no
  `pip install` was run, no environment was modified, no Docker container was touched, no notebook
  was rerun.

## Five-name PyPI PoC

Performed against the real `pypi_retriever.retrieve()` production entry point, contacting the real
PyPI Simple API.

| Import | Distribution | Status | Candidate versions |
|---|---|---|---|
| `sklearn` | `scikit-learn` | `resolved` | `1.7.2`, `1.7.1`, `1.7.0`, `1.6.1`, `1.6.0` |
| `umap` | `umap-learn` | `resolved` | `0.5.12`, `0.5.11`, `0.5.9.post2`, `0.5.8`, `0.5.7` |
| `pkg_resources` | `setuptools` | `resolved` | `84.0.0`, `83.0.0`, `82.0.1`, `82.0.0`, `81.0.0` |
| `scipy` | `scipy` | `resolved` | `1.15.3`, `1.15.2`, `1.15.1`, `1.15.0`, `1.14.1` |
| `dms_variants` | *(no verified mapping)* | `mapping_unknown` | *(none)* |

Confirmed during this check:

- The resolved Python version was `3.10` in every call, sourced from `config/rag_repair.yaml`, not
  derived per import.
- `dms_variants` had `source_endpoint: null` - no PyPI request occurred, confirming the mapping gate
  in `pypi_retriever.retrieve()` (see `docs/rag-design.md` §2.3, step 3) really does stop before any
  network activity for an unmapped import, under real conditions and not just a mocked one.
- Legacy `.exe`, `.egg`, and `.rpm` release artifacts on some of these projects' real PyPI histories
  produced non-fatal warnings (unparseable filenames, per `_group_files_by_version()`'s warning path)
  rather than crashing or being silently miscounted as valid releases.

**Interpretation.** Behavior matched the L5 design: mapping resolution, candidate filtering, status
handling, and the unmapped-import safety gate are the stable findings this check confirms, and they
held under real PyPI data. The **exact candidate version numbers are not a stable finding** - they
will drift every time PyPI publishes a new release for any of these five projects, so they should be
read as "what PyPI reported on 2026-08-17," not as an expected constant. No Ollama call, no
installation, and no notebook execution occurred during this check - it exercises only
`pypi_retriever.py`.

## Three-record pilot

Performed against `scripts/rag_repair_agent.py`'s real `run_repair_agent()`, using a real local
Ollama call for every `usable` record. Output: `data/repair-proposals/i4_live_pilot.jsonl` (4 lines,
committed byte-identically alongside this document - see the commit's final report for the SHA-256
confirming this).

### 1. ID 15 - excluded system-library case (line 0)

`scope_status: excluded`, `refined_subtype: system_library` (`libxcb.so.1` missing). Result:
`status: "abstained"`, `attempts: 0`, `eligibility.decision: "excluded"`,
`retrieval_result: null`, `llm: null`, `final_action: "none"`, `argv: null`. Confirms the
eligibility gate stops a real excluded record before any PyPI or Ollama call under real conditions,
not just a mocked one.

### 2. Initial ID 8 attempt - timeout under the original 120s ceiling (line 1)

`scope_status: usable`, `refined_subtype: missing_package` (`sklearn`). Retrieval succeeded
(`status: "resolved"`, `distribution_name: "scikit-learn"`, 5 compatible candidates). Both Ollama
attempts (the initial call and the one bounded retry) timed out against the then-committed 120-second
`timeout_seconds`. Result: `attempts: 2`, `status: "failed"`, `errors: ["timeout: timed out"]`, no
`proposal`, no `argv`. **This record is preserved exactly as produced** - it is the direct evidence
that justified the timeout change documented below and in `config/rag_repair.yaml`.

### 3. ID 8 rerun - success at 300s (line 2)

Same record, rerun with a temporary local override (`/tmp/rag_repair_pilot.yaml`, outside this
repository and not depended on by any committed code) raising the Ollama timeout to 300 seconds.
Result: `status: "success"`, `attempts: 1`, schema validation `valid: true`, grounding validation
`valid: true`, `proposal: {"action": "install", "install_name": "scikit-learn", "version": null,
"rationale": "..."}`, `argv: ["python", "-m", "pip", "install", "scikit-learn"]`. Ollama was healthy
throughout - the original failure was purely a matter of generation time on CPU-only hardware, not a
connectivity or model-availability problem.

### 4. ID 174 - wrong_version proposal at 300s (line 3)

`scope_status: usable`, `refined_subtype: wrong_version`,
`error_message: "cannot import name 'cumtrapz' from 'scipy.integrate' ..."`. Deterministic extraction
produced `module_path: "scipy.integrate"`, `symbol: "cumtrapz"`. Compatibility evidence resolved to
`compatible_specifier: "<1.14.0"`. Retrieved candidates: `1.13.1`, `1.13.0`, `1.12.0`, `1.11.4`,
`1.11.3` (the intersection of real PyPI's safe set with that evidence constraint). Ollama proposed
`pin_version` / `scipy` / `1.13.1` - schema-valid and grounded on the first attempt. Result:
`attempts: 1`, `status: "success"`,
`argv: ["python", "-m", "pip", "install", "scipy==1.13.1"]`.

## What the pilot proves

- Real PyPI retrieval works end to end against the live API, not just mocked fixtures.
- `gemma2:9b`, run locally, can produce schema-valid, grounded proposals for both supported subtypes.
- Excluded records abstain before any PyPI or Ollama call, under real conditions.
- Timeout and bounded-retry behavior works safely - a slow model produces a clean `failed` result,
  never a hang, a crash, or a fabricated proposal.
- Retrieval fields and full agent results can be persisted to a JSONL file.
- Deterministic argv construction produces a plausible, well-formed pip argument list for both a
  fresh install and a version pin.

## What it does not prove

- No package was installed.
- No Docker environment was modified.
- No notebook was rerun.
- No repair was verified - "success" here means a validated, grounded *proposal*, not a confirmed fix.
- FixApplicator and ResultLogger remain unimplemented; nothing downstream of this pilot's JSONL file
  exists yet.
- This was a 3-record pilot, not the eventual 187-row evaluation split - response reliability, retry
  frequency, and timing at that scale remain unmeasured.

## cv2 / skimage mapping fix validation

Local record of a follow-up check performed after a pre-push review found that `cv2` and `skimage`,
both present as `usable`, `missing_package` rows in the current dependency-error dataset, had no
entry in `config/package_mapping.yaml`, so `pypi_retriever.retrieve()` returned `mapping_unknown` for
them instead of resolving a distribution. `cv2: opencv-python` and `skimage: scikit-image` were added
to `config/package_mapping.yaml`, and the fix was re-checked against the real `retrieve()` production
entry point and the real PyPI Simple API, the same way the five-name PoC above was checked.

- Date: 2026-08-23.

| Import | Distribution | Status | Candidate versions |
|---|---|---|---|
| `cv2` | `opencv-python` | `resolved` | `5.0.0.93`, `4.14.0.94`, `4.13.0.92`, `4.13.0.90`, `4.12.0.88` |
| `skimage` | `scikit-image` | `resolved` | `0.25.2`, `0.25.1`, `0.25.0`, `0.24.0`, `0.23.2` |
| `sklearn` (regression check) | `scikit-learn` | `resolved` | `1.7.2`, `1.7.1`, `1.7.0`, `1.6.1`, `1.6.0` |

Confirmed during this check:

- The real dataset row for `cv2` (`notebook_execution_id: 79`, `scope_status: usable`,
  `refined_subtype: missing_package`) now resolves `distribution_name: "opencv-python"` via
  `resolve_distribution_name()` instead of stopping at `mapping_unknown` - the failure mode a
  pre-push review had flagged.
- `sklearn -> scikit-learn` (an existing, previously-verified mapping) still resolves correctly,
  confirming the new entries did not disturb existing mapping behavior.
- As with `scikit-learn` in the original five-name PoC, legacy `.exe`-based release artifacts on
  `scikit-learn`'s and `opencv-python`'s real PyPI histories produced non-fatal warnings
  (unparseable filenames) rather than being miscounted as valid releases.
- The exact candidate version numbers reflect what PyPI reported on 2026-08-23 and will drift as
  these projects publish new releases - read them as a point-in-time confirmation, not a constant.

No Ollama call, no installation, and no notebook execution occurred during this check - it exercises
only `pypi_retriever.py`, exactly as the five-name PoC above did.

## Issue #42 traceability (local record only - GitLab itself was not modified)

Live evidence gathered above locally supports the following checklist items; this is a note for
whoever next updates the actual GitLab issue, not an update to it:

- **Item 7** ("Persist retrieval fields... per the l5 schema") - `data/repair-proposals/i4_live_pilot.jsonl`
  now contains real, persisted retrieval fields (`distribution_name`, `source_endpoint`,
  `candidate_versions`, `status`, `compatibility_evidence`) for both successful pilot records, not
  only the schema capability described in prior documentation.
- **Item 9** ("Re-run the l5 proof-of-concept cases... against the production module") - all five
  names (`sklearn`, `umap`, `pkg_resources`, `scipy`, `dms_variants`) have now been checked against
  the real production `retrieve()` entry point and real PyPI, per the table above.
- **Item 10** ("Add caching and rate-limit handling") - both were implemented in commit `18121d7` and
  are now additionally corroborated by this pilot not triggering any unexpected throttling or
  duplicate-request behavior against the real API.

Items outside this document's scope: FixApplicator, ResultLogger persistence, and the full
evaluation-split run remain future work, unaffected by this validation.
