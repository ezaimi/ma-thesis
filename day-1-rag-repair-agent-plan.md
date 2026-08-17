# Day 1 Plan — Build the PyPI-Grounded RAGRepairAgent

## 1. Goal for Day 1

Day 1 builds the component that **suggests a safe, evidence-based fix** for a dependency-related notebook failure.

It does **not** install the package, modify the Docker container, or rerun the notebook. Those actions belong to the **FixApplicator**, which starts on Day 2.

By the end of Day 1, the system should be able to receive:

- A failing import name, such as `sklearn`
- The Python version used by the failed notebook
- The classified error subtype, such as `missing_package`

It should then:

1. Map the import name to the correct PyPI distribution.
2. Ask PyPI which releases exist.
3. Remove yanked, unstable, and incompatible releases.
4. Return a grounded fix suggestion in the architecture's defined format.
5. Refuse to generate an install command when the evidence is insufficient.

The intended flow is:

```text
Broken import
    → resolve the real package name
    → retrieve release evidence from PyPI
    → filter unsafe or incompatible releases
    → create a structured fix suggestion
```

---

## 2. Expected Deliverables

Create or complete these files:

```text
config/package_mapping.yaml
scripts/pypi_retriever.py
scripts/rag_repair_agent.py
tests/test_pypi_retriever.py
tests/test_rag_repair_agent.py
```

The responsibilities are:

- `package_mapping.yaml`: known import-name-to-PyPI-distribution mappings.
- `pypi_retriever.py`: package resolution, PyPI communication, validation, and release filtering.
- `rag_repair_agent.py`: converts safe retrieval evidence into the final fix object.
- Test files: verify successful cases, failure cases, filtering, and the no-guessing safety rule.

---

## 3. Step 0 — Clean and Understand the Branch

Confirm the current branch and inspect the working tree:

```bash
git branch --show-current
git status --short
```

Inspect the leftover changes from i3:

```bash
git diff -- scripts/explanation_validator.py
git diff -- scripts/render_explanation_prompt.py
git diff -- scripts/run_llm_explainer.py
```

Decide separately for each file:

- If the change is intentional and useful, commit it separately with an accurate message.
- If it is confirmed accidental or obsolete, restore it:

```bash
git restore scripts/<filename>.py
```

Do not restore a file before understanding its changes.

### Checkpoint

The working tree is clean, or every remaining change has a known purpose.

---

## 4. Step 1 — Review the Existing Design Contracts

Before coding, review:

- `docs/rag-design.md`, especially §3, §5.2, §6, and §7
- `docs/architecture-note.md`, especially the fix object in §6.1
- `data/prompt-tests/l5_pypi_poc_filter_fixture.json`
- `data/prompt-tests/l5_pypi_poc_results.json`
- `tests/test_explanation_validator.py` for the repository's test style

Confirm the exact internal structure expected for `candidate_versions`. Do not create a competing schema if the proof-of-concept files already define one.

### Checkpoint

The implementation fields and status values match the existing design before coding begins.

---

## 5. Step 2 — Add the Package Mapping

Create `config/package_mapping.yaml` with these initial mappings:

```yaml
sklearn: scikit-learn
umap: umap-learn
pkg_resources: setuptools
scipy: scipy
numpy: numpy
pandas: pandas
Bio: biopython
```

Implement:

```python
def resolve_distribution_name(import_name: str) -> str | None:
    ...
```

Rules:

- Return only names explicitly present in the mapping.
- Do not assume an import and a distribution have the same name.
- Preserve import-name case during lookup because names such as `Bio` are case-sensitive in Python.
- Return `None` when the mapping is unknown.

### Checkpoint

```python
resolve_distribution_name("sklearn") == "scikit-learn"
resolve_distribution_name("Bio") == "biopython"
resolve_distribution_name("dms_variants") is None
```

An unknown mapping must stop processing before any network request.

---

## 6. Step 3 — Implement PEP 503 Name Normalization

Implement:

```python
def normalize_distribution_name(distribution_name: str) -> str:
    ...
```

The function must:

1. Convert the distribution name to lowercase.
2. Replace every run of `.`, `_`, or `-` with one `-`.

Recommended implementation:

```python
re.sub(r"[-_.]+", "-", distribution_name).lower()
```

Examples:

```python
normalize_distribution_name("scikit_learn") == "scikit-learn"
normalize_distribution_name("My.Package_Name") == "my-package-name"
```

### Checkpoint

Every PyPI endpoint is constructed using the normalized distribution name.

---

## 7. Step 4 — Implement the PyPI Simple API Client

Implement a function such as:

```python
def fetch_pypi_project(
    distribution_name: str,
    timeout: float = 10.0,
) -> dict:
    ...
```

Use this endpoint:

```text
https://pypi.org/simple/{normalized_distribution}/
```

The request must contain:

```http
Accept: application/vnd.pypi.simple.v1+json
```

Use `urllib.request.Request` and `urllib.request.urlopen` so no new HTTP dependency is introduced.

Handle these outcomes explicitly:

| Situation | Result status |
|---|---|
| Valid response | Continue processing |
| HTTP 404 | `package_not_found` |
| Timeout, DNS, or connection failure | `network_error` |
| Malformed JSON | `invalid_response` |
| JSON missing required fields | `invalid_response` |

Validate at least that:

- The decoded response is a dictionary.
- The `files` field exists.
- `files` is a list.

Do not catch every exception and label it a network failure. HTTP, connection, parsing, and validation problems should remain distinguishable.

### Checkpoint

- `scikit-learn` returns parsed JSON with a `files` list.
- A nonsense package returns `package_not_found` without throwing.
- The request includes the required JSON `Accept` header.

**Implemented (i4), rate limiting added:** `fetch_pypi_project()` now also throttles real
(uncached) requests to a configurable minimum spacing and handles HTTP 429 with a single bounded,
`Retry-After`-honoring retry - see `docs/rag-design.md` §2.4 for the full contract and
`config/rag_repair.yaml`'s `pypi_client.rate_limit` block for the configured values. The `(status,
data)` return contract from this section's sketch is unchanged; a 429 still resolves to
`"network_error"`, just with a small structured `data` payload instead of `None`.

---

## 8. Step 5 — Group Files by Release Version

The Simple API returns individual distribution files. The code must turn them into one candidate per release version.

Use PEP 440-aware utilities from `packaging`, including:

```python
from packaging.utils import parse_sdist_filename, parse_wheel_filename
from packaging.version import InvalidVersion, Version
```

For each file:

1. Read the filename.
2. Extract the version safely.
3. Add the file metadata to that version's group.
4. Skip or record filenames that cannot be parsed safely.

Do not manually split filenames to extract versions. Package names and version strings can contain separators that make manual parsing unreliable.

### Checkpoint

Multiple wheels and source archives for one version produce one release candidate, not several candidates.

---

## 9. Step 6 — Filter Candidate Releases

Implement a function such as:

```python
def filter_candidate_versions(
    files: list[dict],
    python_version: str | None,
    limit: int = 5,
) -> list[dict]:
    ...
```

Apply the following logic in this order.

### 9.1 Drop yanked releases

PyPI's `yanked` value belongs to individual files.

- If every usable file for a release is yanked, remove the release.
- If at least one usable file is not yanked, evaluate only the non-yanked files.

### 9.2 Drop unstable releases

Use `packaging.version.Version` and remove releases for which either is true:

```python
version.is_prerelease
version.is_devrelease
```

This excludes alpha, beta, release-candidate, and development releases.

Do not identify unstable versions through manual text searches.

### 9.3 Check Python compatibility

Use:

```python
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version
```

When the notebook's Python version is known:

- `compatible`: at least one usable, non-yanked file explicitly supports the Python version.
- `incompatible`: the declared constraints reject the Python version.
- `unknown`: no usable file declares `requires_python`.

Unknown must not be presented as proven compatible. For a safe fix, exclude both incompatible and unknown releases when the Python version is known.

When the Python version is unknown:

- Keep non-yanked stable releases as retrieval information.
- Mark compatibility as unknown.
- Do not claim that any release is compatible.

### 9.4 Sort and limit

Sort using `Version`, newest first, and retain at most five candidates.

Do not sort ordinary version strings because values such as `1.10` and `1.9` may be ordered incorrectly.

If the Python version is known and no safe release survives, return `no_compatible_release`.

### Checkpoint

Run the filter against:

```text
data/prompt-tests/l5_pypi_poc_filter_fixture.json
```

Confirm that it removes the fixture's yanked, incompatible, and prerelease entries while preserving the expected safe entries.

---

## 10. Step 7 — Assemble the Retrieval Result

**Implemented (i4):** `retrieve()` exists in `scripts/pypi_retriever.py` with a signature extended
beyond this section's original sketch - it also accepts `subtype`, `module_path`, and `symbol`, so
it can perform the `wrong_version` compatibility-evidence intersection this plan's Step 8 (below)
originally left to `rag_repair_agent.py`. Calls omitting those three parameters behave exactly as
sketched here. See `docs/rag-design.md` §2.3 for the exact, current input/output contract; that
section is authoritative where it differs from this plan.

Implement the public entry point:

```python
def retrieve(
    import_name: str,
    python_version: str | None = None,
) -> dict:
    ...
```

It must always return the exact retrieval fields:

```python
{
    "status": ...,
    "import_name": ...,
    "distribution_name": ...,
    "package_found": ...,
    "python_version": ...,
    "latest_version": ...,
    "candidate_versions": [],
    "source_endpoint": ...,
    "retrieved_at": ...,
    "error": ...,
}
```

Use:

```python
datetime.now(timezone.utc).isoformat()
```

Supported statuses should include:

```text
resolved
mapping_unknown
package_not_found
network_error
invalid_response
no_compatible_release
```

### Mapping short-circuit

For an unknown mapping, return immediately with:

- `status: "mapping_unknown"`
- `distribution_name: null`
- `package_found: false`
- Empty candidates
- `source_endpoint: null`

No PyPI request may occur.

### Checkpoint

The retrieval function always returns the same top-level schema, including when it fails.

---

## 11. Step 8 — Build the Structured Fix Suggestion

**Implemented (i4), with two deviations from this section's sketch:** (1) the entry point is
`run_repair_agent(record, config)` - it takes the full enriched i2 record and the loaded
`config/rag_repair.yaml`, not the bare `(import_name, python_version, subtype)` shown below,
because it also runs the eligibility gate, signature extraction, and LLM call this section assumed
were already handled elsewhere. (2) the fix suggestion now genuinely goes through an LLM proposal
(`prompts/dependency_repair_v1.txt`) plus a separate deterministic grounding check
(`scripts/repair_proposal_validator.py`), not a single deterministic `suggest_fix()` - the "safe
success/failure behaviour" and the `command`-from-validated-fields invariant below are both still
exactly honored, just split across `rag_repair_agent.py` and `repair_proposal_validator.py`. See
`docs/rag-design.md` §2.5 for the current, authoritative contract.

Implement the RAGRepairAgent entry point in `scripts/rag_repair_agent.py`, for example:

```python
def suggest_fix(
    import_name: str,
    python_version: str | None,
    subtype: str = "missing_package",
) -> dict:
    ...
```

The result must follow the fix contract in `docs/architecture-note.md` §6.1:

```python
{
    "action": ...,
    "import_name": ...,
    "install_name": ...,
    "version": ...,
    "command": ...,
    "rationale": ...,
    "pypi_evidence": {
        "latest_version": ...,
        "chosen_version": ...,
        "requires_python": ...,
    },
}
```

### Safe success behaviour

For a missing package with:

- A known mapping
- A package found on PyPI
- At least one stable release proven compatible with the target Python version

The agent may create a fix suggestion using the mapped distribution name and the chosen compatible version.

The command must be created from validated structured fields, not copied from unconstrained LLM text.

Before finalizing this rule, check the contract's relationship between `action`, `version`, and `command`. If §6.1 requires a plain `install` action to have `version: null`, follow that definition exactly and document how the chosen compatible release is represented in the evidence.

### Safe failure behaviour

For these retrieval statuses:

- `mapping_unknown`
- `package_not_found`
- `network_error`
- `invalid_response`
- `no_compatible_release`

Return a safe no-fix result:

```python
{
    "action": "none",
    "import_name": import_name,
    "install_name": None,
    "version": None,
    "command": None,
    "rationale": "...",
    "pypi_evidence": {...},
}
```

The central invariant is:

```python
if result["action"] == "none":
    assert result["command"] is None
```

For `wrong_version`, PyPI's Python compatibility alone does not prove which historical release contains the API required by the notebook. Do not guess an older version. Return `action: "none"` unless the existing RAG design provides additional grounded evidence for choosing it.

**Update:** that additional grounded evidence now exists for a curated set of patterns. `scripts/compatibility_evidence.py` (`lookup_compatibility_evidence`, `filter_versions_by_compatibility_evidence`) checks `config/api_compatibility_evidence.yaml`, a hand-verified registry of official release-notes/changelog boundaries, and intersects it with PyPI's safe candidate versions. Only versions surviving that intersection may ever be proposed as `pin_version`; everything else still returns `none`. As of this writing the registry covers exactly the three unique `wrong_version` patterns present in the dataset (`scipy.integrate.cumtrapz`, `scipy.sparse.sputils.isshape`, `numpy.VisibleDeprecationWarning` — see `docs/rag-design.md` §2.2). A pattern outside that set still has no grounded evidence and must return `none`, exactly as this section originally specified. The worked examples below now use the authoritative execution-runtime constant, `"3.10"` (`config/rag_repair.yaml`, `runtime.python_version` — see `docs/rag-design.md` §4), not the placeholder `"3.11"` earlier drafts of this checkpoint used.

### Checkpoint

- `sklearn` can produce a mapped, PyPI-grounded suggestion.
- `dms_variants` produces `action: "none"` and no command.

---

## 12. Step 9 — Add Automated Tests

Mirror the style of `tests/test_explanation_validator.py` and mock HTTP communication in unit tests.

### Required retriever tests

1. `sklearn` maps to `scikit-learn`.
2. `dms_variants` returns `mapping_unknown`.
3. An unknown mapping performs no HTTP request.
4. PEP 503 normalization works.
5. The required `Accept` header is present.
6. HTTP 404 becomes `package_not_found`.
7. A connection failure becomes `network_error`.
8. Malformed or incomplete JSON becomes `invalid_response`.
9. The existing fixture's yanked release is removed.
10. Its prerelease is removed.
11. Its incompatible release is removed.
12. Missing `requires_python` is not treated as compatible when Python is known.
13. Candidate versions are sorted correctly.
14. No more than five candidates are returned.

### Required RAGRepairAgent tests

1. A successful result contains every fix-object field.
2. The command uses the mapped PyPI distribution name.
3. `mapping_unknown` produces `action: "none"`.
4. `no_compatible_release` produces no command.
5. Network failure produces no command.
6. No fix command can be produced without sufficient PyPI evidence.

Run the focused tests:

```bash
pytest tests/test_pypi_retriever.py tests/test_rag_repair_agent.py -v
```

Then run the entire suite:

```bash
pytest -v
```

### Checkpoint

All existing and new tests pass.

---

## 13. Step 10 — Run the Known Smoke-Test Cases

Run these six cases:

```text
sklearn
umap
pkg_resources
Bio
scipy
dms_variants
```

Example:

```bash
PYTHONPATH=scripts python -c \
"from pypi_retriever import retrieve; print(retrieve('sklearn', '3.10'))"
```

Expected retrieval outcomes:

| Import name | Distribution name | Expected status |
|---|---|---|
| `sklearn` | `scikit-learn` | `resolved` |
| `umap` | `umap-learn` | `resolved` |
| `pkg_resources` | `setuptools` | `resolved` |
| `Bio` | `biopython` | `resolved` |
| `scipy` | `scipy` | `resolved` |
| `dms_variants` | None | `mapping_unknown` |

Compare the statuses and safety behaviour with:

```text
data/prompt-tests/l5_pypi_poc_results.json
```

Current PyPI versions can differ from the proof-of-concept results, so the outputs do not need to be byte-identical.

Test the final fix suggestion:

```bash
PYTHONPATH=scripts python -c \
"from rag_repair_agent import suggest_fix; print(suggest_fix('sklearn', '3.10'))"
```

Test the unknown-mapping safety path:

```bash
PYTHONPATH=scripts python -c \
"from rag_repair_agent import suggest_fix; print(suggest_fix('dms_variants', '3.10'))"
```

The second result must contain:

```text
action = none
command = None
```

---

## 14. Step 11 — Check Tomorrow's Docker Prerequisites

Do not build the FixApplicator today, but spend a short time confirming that Day 2 is technically possible.

For one real failed notebook, identify:

- Its repository and notebook execution ID
- The Docker image or container used for the failed execution
- Whether the container still exists
- Whether the image can be recreated if necessary
- How a command will be executed inside the environment
- How the full notebook will be rerun
- How the new outcome and error will be captured

This is a risk check only. It prevents discovering a Docker lifecycle blocker after the FixApplicator work begins.

### Checkpoint

There is a clear path for applying one suggested fix and rerunning one real notebook on Day 2.

---

## 15. Step 12 — Review and Commit

Review all changes:

```bash
git status
git diff --stat
git diff
```

Confirm that:

- No unrelated i3 changes are included.
- No generated caches or local environment files are staged.
- The mapping configuration is included.
- Tests describe the implemented behaviour accurately.
- No unsafe retrieval result can generate an install command.

Stage only the intended Day 1 files:

```bash
git add \
  config/package_mapping.yaml \
  scripts/pypi_retriever.py \
  scripts/rag_repair_agent.py \
  tests/test_pypi_retriever.py \
  tests/test_rag_repair_agent.py
```

Commit with an accurate message:

```bash
git commit -m "i4: add PyPI-grounded package retrieval and fix suggestions"
```

Push after the full test suite passes.

---

## 16. Definition of Done for Day 1

Day 1 is complete only when:

- [ ] The branch started from a clean, understood state.
- [ ] Import names resolve only through the mapping table.
- [ ] Unknown mappings never trigger PyPI calls.
- [ ] PyPI receives the required JSON `Accept` header.
- [ ] HTTP, network, parsing, and validation failures are handled safely.
- [ ] Release versions are interpreted with PEP 440-aware tooling.
- [ ] Yanked, unstable, and incompatible releases are excluded.
- [ ] At most five safe candidate versions are returned.
- [ ] The retrieval result matches the §6 retrieval schema.
- [ ] The fix suggestion matches the architecture's §6.1 fix-object schema.
- [ ] Insufficient evidence always produces `action: "none"` and `command: null`.
- [ ] All new and existing tests pass.
- [ ] Five known imports resolve successfully.
- [ ] `dms_variants` returns `mapping_unknown` without network activity.
- [ ] At least one real Docker case has been inspected for Day 2 feasibility.
- [ ] The completed work is reviewed and committed without unrelated files.

---

## 17. What Is Explicitly Not Part of Day 1

Day 1 does not:

- Run `pip install` inside a container.
- Modify a notebook's environment.
- Rerun a notebook.
- Decide whether a proposed fix actually worked.
- Write a `repair_attempts` database row.
- Enrich the knowledge graph.

Those responsibilities belong to later components:

| Component | Purpose | Planned time |
|---|---|---|
| RAGRepairAgent | Decide which grounded fix should be attempted | Day 1 |
| FixApplicator | Apply the command and rerun the notebook | Day 2 |
| ResultLogger | Save the attempt and its outcome | Day 3 |

The Day 1 result is therefore a **safe proposed fix with PyPI evidence**, not a confirmed repair.
