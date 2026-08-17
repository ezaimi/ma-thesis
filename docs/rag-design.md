# L5 — PyPI RAG Design and Schema Definition


## 1. Goal

L5 designs the PyPI-grounded retrieval component used by `RAGRepairAgent`.

Its purpose is to provide verified Python package metadata to the repair prompt so that repair proposals do not guess package names, package versions, or Python-version compatibility.

L5 defines the retrieval design, result schema, import-name resolution policy, candidate filtering, prompt-context format, edge-case behaviour, and persisted retrieval provenance.

L5 includes a small proof of concept only. The production retriever will be implemented later in i4.


## 2. Scope and Boundary

L5 supports dependency-related failures in Python Jupyter notebooks.

The retrieval component uses PyPI as a source of package metadata, including:

- package existence;
- available releases;
- yanked-release status;
- Python-version requirements.

PyPI metadata alone does not prove that a specific imported API exists in a particular package version. For example, PyPI can show that a SciPy release exists and supports a Python version, but it does not prove that a specific function such as `cumtrapz` is available in that release.

When a package mapping, compatible version, or API-level compatibility cannot be verified from available evidence, the repair agent must return `none`.

L5 does not implement the production HTTP retriever, apply repairs, or rerun notebooks. Those activities belong to later implementation and evaluation stages.

### 2.1 Repair-eligibility gate (i4 entry contract)

`RAGRepairAgent` must only be invoked for records where the upstream classification carries
`scope_status == "usable"`. This is a precondition on whether the component runs at all, and is
distinct from the `mapping_unknown` / `package_not_found` / `no_compatible_release` retrieval
statuses in §6 and §9, which describe what the retriever does *after* it has already been invoked
on an eligible record but cannot safely resolve it. Do not confuse the two: `scope_status` is i1's
repair-eligibility classification (`missing_package`/`wrong_version` -> `usable`;
`system_library`/`mapping_unknown` -> `excluded`); L5's own `mapping_unknown` retrieval status is
a narrower, PyPI-resolution-time outcome that can occur even for a `usable` row (e.g. an
unrecognised import name with no verified distribution mapping, such as `dms_variants`).

`LLMExplainer` (O1) has no such gate and explains every `DEPENDENCY_ERROR` row regardless of
`scope_status` - see `docs/architecture-note.md` §7.1 and the "i4 scope clarification" note in
`docs/prompts.md`.

As of this writing, no orchestrator or `RAGRepairAgent` entry point exists yet in the repository
(`scripts/rag_repair_agent.py` is planned but not yet implemented; `scripts/pypi_retriever.py`
currently implements only import-name mapping resolution). When the entry point is built, it -
or the orchestrator calling it - must check `scope_status == "usable"` before calling it, using
the field now carried through `data/context-classification/dependency_error_contexts.jsonl` (i2)
and `data/llm-explanations/explanation_results.jsonl`'s `input.scope_status` (i3). Excluded
records should still produce a `repair_attempts`-style outcome value that distinguishes "repair
was never attempted because the record is out of the pip-only scope" from "repair was attempted
and returned `none`" - the exact field name and value are left to the i4/O4 implementation, since
the `repair_attempts` table (§6.2 of `docs/architecture-note.md`) does not yet define one.



## 3. Import Name Resolution Policy

Python import names and PyPI distribution names are not always identical.

For example:

| Python import name | Verified PyPI distribution name |
| --- | --- |
| `sklearn` | `scikit-learn` |
| `umap` | `umap-learn` |
| `pkg_resources` | `setuptools` |
| `scipy` | `scipy` |
| `numpy` | `numpy` |
| `pandas` | `pandas` |
| `Bio` | `biopython` |

The verified mapping table includes both renamed mappings and curated identity mappings. An identity mapping is valid only when it is explicitly listed in the table; the retriever must never assume that every import name is also the PyPI distribution name.

The retriever must resolve the import name before querying PyPI.

Resolution follows this order:

1. Check a verified import-to-distribution mapping table.
2. If a verified mapping exists, query PyPI using the resolved distribution name.
3. If no verified mapping exists, return the retrieval status `mapping_unknown`.
4. Do not assume that an import name is also a pip-installable package name.

For example, `dms_variants` must return `mapping_unknown` unless a verified mapping is available. The system must not automatically propose `pip install dms_variants`.

A `mapping_unknown` result provides no install command and no candidate version. The repair agent must return `none` unless later evidence verifies the distribution name.

In this document, `distribution_name` means the verified PyPI distribution name used for retrieval. When the repair agent proposes `install` or `pin_version`, this same value is written to the `install_name` field defined in the L4 repair schema. For other outcomes, such as `package_not_found` or `no_compatible_release`, `distribution_name` may still be recorded even though the repair agent returns `none` and `install_name` remains `null`.

## 4. Retrieval Inputs

The retriever receives structured context from the failed notebook execution and earlier pipeline stages.

| Input | Required | Purpose |
| --- | --- | --- |
| `import_name` | yes | Python import involved in the failure, for example `sklearn`. |
| `distribution_name` | no | Verified PyPI distribution name after name resolution, for example `scikit-learn`. |
| `error_type` | yes | Dependency-related Python error type, such as `ModuleNotFoundError` or `ImportError`. |
| `traceback` | yes | Error message or traceback used to identify the dependency problem. |
| `python_version` | no | Python runtime version of the failed notebook environment, used to filter incompatible releases. |
| `installed_version` | no | Known installed version of the affected distribution, if available. |
| `current_requirements` | no | Existing dependency declarations from the repository or notebook environment. |
| `prior_attempt` | no | Previous repair result, used only in a later repair round. |

The minimum retrieval input is `import_name`, `error_type`, and `traceback`.

If `distribution_name` cannot be resolved through the verified mapping policy, the retriever returns `mapping_unknown` and does not query PyPI.

If `python_version` is unavailable, the retriever may return package metadata but must not claim that a candidate release is compatible with the notebook runtime.




## 5. PyPI Endpoint Strategy

The retriever uses official, read-only PyPI endpoints only after a Python import name has been resolved to a verified distribution name.

### 5.1 Relation to PLLM [5]

PLLM [5] motivates the use of PyPI-grounded metadata and a curated import-to-install-name mapping to reduce unsupported dependency and version guesses.

L5 adopts these goals. However, its candidate-version discovery uses the PyPI JSON Simple API rather than relying on the `releases` field of the project JSON endpoint. This is an intentional implementation update: PyPI recommends the JSON version of its Index API for new integrations, and its JSON API documentation marks the `releases` field as deprecated.[^pypi-index][^pypi-json]

The project and version-specific JSON endpoints remain available for supplementary metadata when required.

### 5.2 Candidate-Version Discovery

The primary endpoint for discovering available package versions is the PyPI JSON Simple API:

```text
GET /simple/{distribution}/
Accept: application/vnd.pypi.simple.v1+json
```

Example:

```text
GET https://pypi.org/simple/scikit-learn/
```

Before querying the Simple API, the verified distribution name is normalized according to PEP 503: it is lowercased, and consecutive `.`, `_`, or `-` characters are replaced with one `-`.

The JSON Simple API provides information needed for candidate-version filtering, including:

- available package versions;
- distribution files;
- `requires-python` metadata;
- yanked-release status;
- upload time;
- file URL and hash metadata.

This endpoint is the primary source for finding and filtering candidate package versions.

### 5.3 Project Metadata

The project JSON endpoint is:

```text
GET /pypi/{distribution}/json
```

Example:

```text
GET https://pypi.org/pypi/scikit-learn/json
```

It may be used to retrieve high-level project metadata, such as:

- canonical project name;
- latest version;
- latest `requires_python` value;
- project URLs.

The retriever must not rely on the `releases` field from this endpoint for candidate-version selection.

### 5.4 Version-Specific Metadata

When additional metadata is needed for one selected candidate version, the retriever may use:

```text
GET /pypi/{distribution}/{version}/json
```

Example:

```text
GET https://pypi.org/pypi/scipy/1.13.1/json
```

This endpoint is used only after a candidate version has already been selected or needs additional validation.

### 5.5 Endpoint Usage Order

The intended retrieval flow is:

```text
import_name
    ↓
verified import-to-distribution mapping
    ↓
GET /simple/{distribution}/ with JSON response
    ↓
filter candidate versions
    ↓
optional project or version-specific metadata request
    ↓
build compact prompt context
```

For example:

```text
sklearn
    ↓
verified mapping: scikit-learn
    ↓
query PyPI for scikit-learn metadata
    ↓
filter versions based on Python compatibility and yanked status
    ↓
provide safe package evidence to the repair prompt
```

If the import name cannot be resolved through the verified mapping policy, the retriever returns `mapping_unknown` before making any PyPI request.


## 6. Retrieval-Result Schema

The retriever returns one structured result for each dependency-related error.

The result records whether the import name was resolved, whether PyPI metadata was retrieved successfully, and which package versions are safe candidates for the repair prompt.

```json
{
  "status": "resolved | mapping_unknown | package_not_found | no_compatible_release | network_error | invalid_response",
  "import_name": "string",
  "distribution_name": "string or null",
  "package_found": "boolean or null",
  "python_version": "string or null",
  "latest_version": "string or null",
  "candidate_versions": [
    {
      "version": "string",
      "requires_python": "string or null",
      "python_compatibility": "compatible | incompatible | unknown",
      "yanked": "boolean",
      "yanked_reason": "string or null"
    }
  ],
  "source_endpoint": "string or null",
  "retrieved_at": "ISO 8601 timestamp or null",
  "error": "string or null"
}
```

### Field meanings

The fields `requires_python`, `python_compatibility`, `yanked`, and `yanked_reason` belong to each item inside `candidate_versions`; they are not top-level retrieval-result fields.

| Field | Meaning |
| --- | --- |
| `status` | Final retrieval outcome. |
| `import_name` | Python import that caused the failure, for example `sklearn`. |
| `distribution_name` | Verified PyPI distribution name used for retrieval, for example `scikit-learn`. It is `null` when the mapping is unknown. |
| `package_found` | Whether the resolved distribution was found on PyPI. It is `null` when no PyPI query was made. |
| `python_version` | Python runtime version of the failed notebook, if known. |
| `latest_version` | Latest available package version reported by PyPI, if retrieval succeeds. |
| `candidate_versions` | Bounded list of versions considered relevant after filtering. |
| `requires_python` | Python-version requirement declared for that candidate release. |
| `python_compatibility` | Whether the candidate is compatible with the known notebook Python version. It is `unknown` if the runtime version is unavailable. |
| `yanked` | Whether the release was withdrawn on PyPI. |
| `yanked_reason` | PyPI’s stated reason for withdrawing the release, if available. |
| `source_endpoint` | PyPI endpoint used to obtain the result. |
| `retrieved_at` | Time at which the metadata was retrieved. |
| `error` | Retrieval or parsing error details, if applicable. |

### Schema rules

- `mapping_unknown` means that no verified import-to-distribution mapping exists. In this case, `distribution_name`, `package_found`, `latest_version`, `candidate_versions`, `source_endpoint`, and `retrieved_at` are `null` or empty.
- `package_not_found` means that a distribution name was resolved but PyPI did not contain that distribution.
- `no_compatible_release` means that PyPI returned package metadata but no non-yanked release was compatible with the known Python runtime.
- `network_error` means that PyPI could not be reached or did not respond within the configured timeout.
- `invalid_response` means that PyPI returned a response that could not be parsed or did not contain the expected metadata.
- A retrieval result must not claim that a specific imported function or API exists in a selected version unless separate compatibility evidence is available.
- The retriever does not return an install command. It returns only verified metadata for the repair prompt.



## 7. Candidate-Version Filtering

Candidate-version filtering reduces the full set of PyPI releases to a small, safe, and relevant set of metadata entries for the repair prompt.

The retriever performs filtering only when the retrieval status is `resolved` and the distribution was found on PyPI.

### 7.1 Filtering rules

The retriever must:

1. Group PyPI files by package version so that one candidate represents one release version.
2. Exclude yanked releases by default.
3. Exclude pre-release, development, and release-candidate versions by default.
4. When `python_version` is known, exclude versions whose declared `requires_python` value is incompatible with that runtime.
5. When `python_version` is unavailable, retain non-yanked stable versions but mark their Python compatibility as `unknown`.
6. Keep only a bounded number of relevant versions, sorted from newest to oldest.
7. Return at most five candidate versions to avoid inserting excessive release metadata into the repair prompt.

If a distribution has only pre-release, development, or release-candidate versions, v1 returns `no_compatible_release`. Selecting pre-release versions is outside the v1 repair scope.

### 7.2 Candidate selection order

The retriever selects candidates in this order:

```text
all available PyPI release files
    ↓
group files by package version
    ↓
remove yanked releases
    ↓
remove pre-release and development versions
    ↓
filter by known Python-version compatibility
    ↓
sort newest to oldest
    ↓
keep at most five versions
```

### 7.3 Python-version compatibility

The retriever evaluates `requires_python` against the notebook runtime version when that version is available.

For example:

```text
Notebook Python version: 3.10
Package requirement: >=3.9
Result: compatible
```

```text
Notebook Python version: 3.8
Package requirement: >=3.9
Result: incompatible
```

If the notebook Python version is unknown, the retriever must not claim that a release is compatible. It records the compatibility value as `unknown`.

If a candidate release does not declare `requires_python`, the retriever cannot evaluate its compatibility even when the notebook's Python version is known. Such a release is recorded as `python_compatibility: unknown`, using the same convention as when the notebook's Python version itself is unavailable.

### 7.4 Safety boundary

Candidate versions are metadata, not repair instructions.

The retriever must not claim that a candidate version contains a specific missing function or API. For example, PyPI metadata can show that SciPy `1.13.1` exists and supports a Python version, but it cannot prove that `scipy.integrate.cumtrapz` is available in that version.

A `pin_version` repair requires separate API-level compatibility evidence in addition to PyPI release metadata.

The retriever also does not verify operating-system, processor, or wheel availability. Those checks remain the responsibility of the later installation and notebook-execution stages.



## 8. Prompt Context Format

The retriever converts its structured retrieval result into a compact text block for the `pypi_versions` input slot of the L4 repair prompt.

The prompt context must clearly distinguish verified metadata from unavailable information. It must not claim API-level compatibility unless separate evidence is provided.

### 8.1 Resolved package context

For a successfully resolved package, the retriever provides:

```text
PyPI retrieval status: resolved
Python import name: sklearn
Resolved distribution name: scikit-learn
Notebook Python version: 3.10
Latest available version: 1.7.0

Candidate versions:
- 1.7.0 | requires_python: >=3.10 | python_compatibility: compatible | yanked: false
- 1.6.1 | requires_python: >=3.9 | python_compatibility: compatible | yanked: false

API-level compatibility evidence:
Not available
```

The repair agent may use the resolved distribution name and candidate-version metadata, but it must not claim that a specific package version fixes an import or API error unless API-level compatibility evidence is also supplied.

### 8.2 Unknown mapping context

When no verified import-to-distribution mapping exists, the retriever provides:

```text
PyPI retrieval status: mapping_unknown
Python import name: dms_variants
Resolved distribution name: Not available
PyPI query: Not performed
Reason: No verified import-to-distribution mapping is available.

Candidate versions:
Not available

API-level compatibility evidence:
Not available
```

For this status, the repair agent must return `none` and must not propose an install command.

### 8.3 Package-not-found context

When a verified distribution name is not found on PyPI, the retriever provides:

```text
PyPI retrieval status: package_not_found
Python import name: example_import
Resolved distribution name: example-package
PyPI query: Completed
Reason: The resolved distribution was not found on PyPI.

Candidate versions:
Not available

API-level compatibility evidence:
Not available
```

### 8.4 Context requirements

The generated prompt context must:

- include the retrieval status;
- include the resolved distribution name only when verified;
- include only bounded, filtered candidate versions;
- clearly show whether Python compatibility is known, incompatible, or unknown;
- state when no PyPI query was performed;
- state when API-level compatibility evidence is unavailable;
- contain no generated install command or unsupported repair claim.



## 9. Edge Cases and Retrieval Statuses

The retriever returns one explicit status for every lookup attempt. A status describes what happened during import-name resolution and PyPI metadata retrieval.

| Status | Meaning | PyPI query performed? | Repair-agent behaviour |
| --- | --- | --- | --- |
| `resolved` | A verified distribution name was resolved and relevant PyPI metadata was retrieved. | Yes | May consider `install` or `pin_version` only when the available evidence supports it. |
| `mapping_unknown` | No verified mapping exists from the Python import name to a PyPI distribution name. | No | Return `none`. Do not invent a package name or command. |
| `package_not_found` | A verified distribution name was resolved, but PyPI did not contain that distribution. | Yes | Return `none`. |
| `no_compatible_release` | Package metadata was retrieved, but no non-yanked stable release was compatible with the known Python runtime. | Yes | Return `none`. |
| `network_error` | PyPI could not be reached, timed out, or returned an HTTP retrieval failure. | Possibly incomplete | Return `none`. |
| `invalid_response` | A PyPI response was received but could not be parsed or lacked expected metadata. | Yes | Return `none`. |

### 9.1 Safe behaviour by status

The retriever must follow these rules:

- Only `resolved` may provide package metadata and bounded candidate versions to the repair prompt.
- All other statuses provide no install command and no pinned-version candidate.
- A failed or incomplete lookup must not stop the notebook-processing batch.
- Retrieval failures must be recorded for later analysis.
- The repair agent must treat unavailable, incomplete, or unverified metadata as insufficient evidence.

### 9.2 Important edge cases

| Edge case | Required behaviour |
| --- | --- |
| Import name has no verified mapping | Return `mapping_unknown` before any PyPI request. |
| Resolved distribution is missing from PyPI | Return `package_not_found`. |
| PyPI service is unavailable or times out | Return `network_error`. |
| PyPI response is malformed or incomplete | Return `invalid_response`. |
| Notebook Python version is unknown | Keep non-yanked stable candidates, but mark Python compatibility as `unknown`. |
| All available releases are yanked | Return `no_compatible_release`. |
| All stable releases conflict with the known Python version | Return `no_compatible_release`. |
| API-level compatibility is not verified | Do not propose `pin_version` based on PyPI metadata alone. |

### 9.3 Retry policy

The L5 design does not define automatic retries for PyPI retrieval.

The production implementation in i4 may use a bounded retry policy for temporary network failures. Any retry policy must:

- use a limited number of attempts;
- record the final retrieval status;
- avoid delaying or crashing the overall batch pipeline;
- never replace a failed retrieval with guessed metadata.



## 10. Persisted Retrieval Provenance

The full retrieval-result schema defined in Section 6 is used while the retriever processes one dependency error.

For later evaluation, debugging, and reproducibility, the system persists a selected subset of those fields as retrieval provenance.

The persisted provenance record does not need to store the complete raw PyPI response.

### 10.1 Fields to persist

| Field | Purpose |
| --- | --- |
| `import_name` | Records the Python import involved in the failure. |
| `distribution_name` | Records the verified PyPI distribution name, if resolved. |
| `status` | Records the final retrieval outcome, such as `resolved` or `mapping_unknown`. |
| `python_version` | Records the notebook runtime version used for compatibility filtering, if known. |
| `source_endpoint` | Records the PyPI endpoint used during retrieval. |
| `retrieved_at` | Records when the metadata was retrieved. |
| `candidate_versions` | Records the bounded filtered candidate versions provided to the repair prompt. |
| `error` | Records retrieval or parsing failure details, if applicable. |

### 10.2 Persistence rules

- Persist only the bounded candidate-version list, not every historical package release.
- Do not persist the complete raw PyPI response unless later evaluation requires it.
- Preserve the retrieval status even when no PyPI request was made.
- For `mapping_unknown`, persist the original import name and status, but leave `distribution_name` and endpoint fields empty.
- For retrieval failures, persist a concise error description without secrets, credentials, or unrelated environment data.
- The persisted record must allow a later evaluator to understand which metadata was available to the repair prompt at the time of generation.

### 10.3 Relationship to later evaluation

Persisted retrieval provenance supports later analysis of:

- whether a repair proposal was based on verified package metadata;
- whether a failure occurred during name resolution or PyPI retrieval;
- which candidate versions were available to the repair agent;
- whether package metadata may have changed since the repair attempt.

The exact database table and implementation details are deferred to the later i4 and O6 implementation stages.



## 11. Proof of Concept Plan

The L5 proof of concept verifies that the proposed retrieval schema and endpoint strategy can provide the metadata required by the later repair module.

It is a small throwaway validation script only. It does not implement the production retriever, modify notebooks, generate repair commands, or apply repairs.

### 11.1 Objective

The proof of concept must confirm that:

- verified import-to-distribution mappings are resolved before any PyPI request;
- PyPI responses provide the fields required by the retrieval-result schema;
- candidate versions can be filtered using yanked status and `requires_python`;
- unresolved imports return `mapping_unknown` without querying PyPI;
- unsafe or incomplete metadata does not produce a version recommendation or install command.

### 11.2 Test cases

| Case | Expected resolution status | Expected behaviour |
| --- | --- | --- |
| `sklearn` | `resolved` | Resolve to `scikit-learn` and retrieve package metadata. |
| `umap` | `resolved` | Resolve to `umap-learn` and retrieve package metadata. |
| `pkg_resources` | `resolved` | Resolve to `setuptools` and retrieve package metadata. |
| `Bio` | `resolved` | Resolve to `biopython` and retrieve package metadata. |
| `scipy` | `resolved` | Retrieve release metadata and verify candidate filtering fields. |
| `dms_variants` | `mapping_unknown` | Do not query PyPI and provide no candidate versions. |

### 11.3 Procedure

For each resolved test case, the throwaway script will:

1. apply the verified import-to-distribution mapping;
2. request PyPI metadata using the endpoint strategy from Section 5;
3. parse the package name, available versions, `requires_python`, and yanked-release fields;
4. create a retrieval result matching the schema in Section 6;
5. filter candidates using the rules in Section 7;
6. print or save the resulting structured metadata for manual review.

For `dms_variants`, the script will confirm that the result is `mapping_unknown` before any PyPI request is made.

### 11.4 Success criteria

The proof of concept succeeds when:

- all resolved cases return a structured retrieval result;
- the result includes a verified distribution name, retrieval status, source endpoint, retrieval timestamp, and filtered candidate versions;
- unresolved imports return `mapping_unknown`;
- yanked or Python-incompatible releases are not included as safe candidates;
- no install command, pinned version, or API-level compatibility claim is generated.

### 11.5 Recorded output

The proof-of-concept output will record:

- input import name;
- resolved distribution name, if available;
- retrieval status;
- source endpoint;
- retrieval timestamp;
- bounded candidate-version metadata;
- errors, if any.

The output will be used to validate the L5 design. Production retrieval, retry handling, database persistence, and integration with `RAGRepairAgent` remain deferred to i4.



## 12. Proof of Concept Results

The L5 proof of concept was executed locally using Python `3.13`.

It used the PyPI JSON Simple API and produced retrieval results matching the schema defined in Section 6.

| Import name | Expected status | Result |
| --- | --- | --- |
| `sklearn` | `resolved` | Resolved to `scikit-learn` and returned filtered candidate versions. |
| `umap` | `resolved` | Resolved to `umap-learn` and returned filtered candidate versions. |
| `pkg_resources` | `resolved` | Resolved to `setuptools` and returned filtered candidate versions. |
| `Bio` | `resolved` | Resolved to `biopython` and returned filtered candidate versions. |
| `scipy` | `resolved` | Resolved to `scipy` and returned filtered candidate versions. |
| `dms_variants` | `mapping_unknown` | No PyPI request was made and no candidate versions were returned. |

Results for all six cases are recorded in `data/prompt-tests/l5_pypi_poc_results.json`.

The proof of concept confirmed that verified mappings are applied before PyPI retrieval and that unresolved imports do not trigger guessed package-name lookups.

For resolved packages, the result included the distribution name, retrieval status, Python version used for filtering, latest available version, bounded candidate versions, source endpoint, and retrieval timestamp.

Some older releases did not declare a `requires_python` value. These releases were retained with `python_compatibility: unknown`; they were not claimed to be compatible. This case was not anticipated when Section 7.3 was first written; the rule has since been added there.

None of the six live PyPI lookups happened to return a yanked or Python-incompatible release, so the exclusion rules in Section 7.1 were not exercised by live data alone. A synthetic fixture (`data/prompt-tests/l5_pypi_poc_filter_fixture.json`), modelled on a real PyPI Simple API response, was used to confirm this separately: of six fixture releases, a yanked release, an incompatible release, and a pre-release were each correctly excluded, while compatible releases and one release with no declared `requires_python` were correctly retained. This also confirms that a release excluded for incompatibility never appears in `candidate_versions` tagged `python_compatibility: incompatible` — under the Section 7.1 rules, incompatible releases are dropped outright rather than retained with that label, so `incompatible` describes an intermediate filtering decision, not a value that survives into the persisted result.

This proof of concept validates the L5 retrieval design only. The Python `3.13` value is the local proof-of-concept runtime and does not represent the runtime of the failed Docker notebooks. The production retriever will later use the actual notebook-container Python version when it is available.

The retrieved versions are metadata only. They do not prove that a version fixes a specific missing API or import. Production retrieval, retry handling, database persistence, repair generation, and notebook re-execution remain deferred to later stages.







[^pypi-index]: PyPI Index API documentation: https://docs.pypi.org/api/index-api/
[^pypi-json]: PyPI JSON API documentation: https://docs.pypi.org/api/json/