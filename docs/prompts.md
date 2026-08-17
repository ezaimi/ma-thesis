# L4 — Open LLM Selection and Prompt Design

## 1. Goal

L4 selects the initial open LLM configuration and designs the explanation and repair prompt templates for dependency-related Jupyter notebook failures.

L4 produces:

- an initial model and runtime decision;
- prompt templates for explanation and repair;
- a small development sample;
- prompt examples and design-time sanity checks.

L4 does not perform the later quantitative model comparison or repair-success evaluation. Those require the FixApplicator and belong to later evaluation steps.

## 2. Scope

The L4 target consists of Docker-pipeline notebook executions classified as `DEPENDENCY_ERROR`.

The v1 repair scope is limited to pip-based dependency actions:

- `install`
- `pin_version`
- `none`

The following are outside the v1 repair scope:

- notebook source-code edits;
- import-path corrections;
- uninstalling conflicting packages;
- system-package installation;
- missing shared libraries such as `.so` files.

The explanation component and repair component remain separate:

- `LLMExplainer` produces a plain-language explanation;
- `RAGRepairAgent` produces a structured repair proposal.

The pip-only v1 scope listed above governs `RAGRepairAgent` only. `LLMExplainer` is not
restricted to this scope: it explains every `DEPENDENCY_ERROR` row, including system-library and
ambiguous-local-module cases that `RAGRepairAgent` will never attempt to repair. See "i4 scope
clarification" at the end of this document and `docs/architecture-note.md` §7.1.

## 3. Model Candidates

This thesis shortlists open LLMs for the L4 prompt-design stage. The models are selected for two tasks:

1. generating plain-language explanations of dependency-related notebook failures;
2. generating structured repair suggestions for pip-installable dependency errors.

The shortlist is grounded in the L2 literature review, especially PLLM [5], Szalontai et al. [9], and the empirical APR model-comparison study [N5].

### Primary model

**Gemma-2 9B**

Gemma-2 9B is selected as the primary candidate because PLLM [5], the closest related work to this thesis, identifies Gemma-2 9B with RAG as its best-performing configuration for Python dependency repair.

This makes Gemma-2 9B the cleanest primary choice because the planned `RAGRepairAgent` also uses PyPI-grounded retrieval.

Gemma-2 9B is the default model for both components:

- `LLMExplainer`
- `RAGRepairAgent`

Using one model for both steps keeps the later e6 baseline clean because explanation and repair results are not affected by mixing model effects.

### Secondary and fallback models

**CodeLlama-13B**

CodeLlama-13B is kept as a fallback candidate because code-specialised models are strongly supported in the automated program repair literature. Szalontai et al. [9] motivate CodeLlama as a repair-oriented model family, and [N5] confirms that code-specialised models are competitive for repair tasks.

**DeepSeek-Coder 7B / 13B**

DeepSeek-Coder 7B and 13B are retained as secondary candidates because [N5] reports strong performance for DeepSeek-Coder in automated program repair, including Python-related repair settings.

The 7B-class model is the preferred DeepSeek fallback under limited hardware. A 13B-class variant is considered only if a compatible local model artifact is available and the hardware can run it reliably.

### L4 decision

The L4 model shortlist is:

| Role       | Model              | Reason                                                 |
| ---------- | ------------------ | ------------------------------------------------------ |
| Primary    | Gemma-2 9B         | Best PLLM + RAG configuration; clean baseline          |
| Fallback 1 | CodeLlama-13B      | Code-specialised repair model                          |
| Fallback 2 | DeepSeek-Coder 7B  | Smaller code-specialised fallback for limited hardware |
| Fallback 3 | DeepSeek-Coder 13B | Higher-capacity code-specialised fallback if available |

The default L4 decision is to use **Gemma-2 9B** for both explanation and repair generation. CodeLlama-13B and DeepSeek-Coder 7B/13B remain fallback candidates.

This is a design shortlist only. L4 does not claim final repair-success superiority. Quantitative model comparison is deferred until the FixApplicator and e2/e3 evaluation steps are available.

## 4. Runtime and Quantisation

The L4 prompt-design stage uses a local open-model serving setup.

### Selected runtime

The selected runtime is:

```text
Ollama
```

Ollama is selected because L4 requires local prompt sanity checks rather than high-throughput production serving. It provides a simple local interface for downloading, running, and inspecting open models.

### Selected quantisation approach

The target quantisation approach is:

```text
4-bit quantised local model
```

The shortlisted models are in the 7B–13B range. Four-bit quantisation reduces memory requirements and makes local testing more realistic on limited hardware.

The exact quantisation level depends on the selected local model artifact. It must be verified after pulling the model and must not be assumed from the model family name alone.

### Alternative runtimes considered

| Runtime          | Decision | Reason                                                                    |
| ---------------- | -------- | ------------------------------------------------------------------------- |
| Ollama           | selected | Simplest setup for local L4 prompt testing                                |
| llama.cpp / GGUF | fallback | Useful if direct GGUF and quantisation control is needed                  |
| vLLM             | deferred | Better suited to high-throughput GPU serving, which is unnecessary for L4 |

### Hardware verification plan

Before prompt sanity checks, the selected candidate models must be tested on the available machine. The following checks will be run first:

```bash
ollama --version
nvidia-smi || true
free -h
```

The primary model will then be tested:

```bash
ollama pull gemma2:9b
ollama run gemma2:9b "Reply only with OK."
ollama ps
ollama show gemma2:9b --verbose
```

The same verification will be repeated for the selected fallback candidate:

```bash
ollama pull codellama:13b
ollama run codellama:13b "Reply only with OK."
ollama ps
ollama show codellama:13b --verbose
```

The exact available DeepSeek-Coder model tag will be selected and verified before testing.

The final L4 record will state:

- the model tag used;
- whether it ran on GPU or CPU;
- the reported quantisation level;
- whether the model completed the basic prompt test reliably.

### Hardware verification result

Gemma-2 9B was tested locally through Ollama on Windows.

```text
Model: gemma2:9b
Memory used: 6.3 GB
Processor: 100% CPU
Context: 4096 tokens
```

### L4 decision

The L4 default serving setup is:

```text
Runtime: Ollama
Quantisation target: 4-bit local quantisation
Primary model to test first: Gemma-2 9B
Fallback under limited hardware: DeepSeek-Coder 7B-class model
Fallback with more available memory: CodeLlama-13B or DeepSeek-Coder 13B
```

This decision is for prompt design and sanity checking. Runtime performance and quantitative model comparison are deferred to the later evaluation phase.

## 5. Model Assignment Decision

The architecture allows model configuration per component. Therefore, the system can use either one shared model or separate models for `LLMExplainer` and `RAGRepairAgent`.

### Decision

The L4 default is to use one shared model:

```text
LLMExplainer: Gemma-2 9B
RAGRepairAgent: Gemma-2 9B
```

Gemma-2 9B is selected for both components because PLLM [5] identifies Gemma-2 9B with RAG as its best-performing configuration for Python dependency repair.

Using one model for both components keeps the later e6 baseline clean. The evaluation can attribute results to one model configuration instead of mixing an explanation model with a different repair model.

### Fallback option

A split-model setup remains possible if prompt sanity checks show weak structured repair output from Gemma-2 9B.

Fallback split setup:

```text
LLMExplainer: Gemma-2 9B
RAGRepairAgent: CodeLlama-13B or DeepSeek-Coder 7B/13B
```

This fallback is justified because CodeLlama and DeepSeek-Coder are code-specialised models supported by the L2 automated-program-repair literature.

### L4 decision

| Component      | Default model | Fallback model                         |
| -------------- | ------------- | -------------------------------------- |
| LLMExplainer   | Gemma-2 9B    | Gemma-2 9B                             |
| RAGRepairAgent | Gemma-2 9B    | CodeLlama-13B or DeepSeek-Coder 7B/13B |

The split-model setup is not the default. It remains available because the architecture supports per-component model configuration.

## 6. Development Sample

A small development sample was created for L4 prompt design and sanity checking.

The sample is stored in:

```text
data/prompt-tests/dev_sample.csv
```

It contains 13 real dependency-error rows selected from the Docker pipeline database:

- 10 `missing_package` rows;
- 3 `wrong_version` rows.

The rows originate from `notebook_executions` where:

```sql
error_category = 'DEPENDENCY_ERROR'
```

System-library errors were excluded because the v1 repair scope is limited to pip-based actions.

Potential namespace-collision cases were also excluded. For example, generic imports such as `utils` or the Python standard-library name `statistics` may be shadowed by an installed package or a local project module. Their likely fixes require uninstalling a package or changing import-path priority, which is outside the v1 action set.

The development sample is used only to design and manually review prompts. It remains separate from the final evaluation set.

## 7. Explanation Prompt

### Purpose

The explanation prompt is used by `LLMExplainer` to explain a dependency-related notebook failure in plain language for a non-expert user.

It is explanation-only. It must not suggest a repair, package version, or command. Repair generation is handled separately by `RAGRepairAgent`.

### Input slots

The prompt receives:

- `error_type`
- `traceback`
- `failing_module`
- `cell_source`

The fields map directly to the L4 development sample. If `cell_source` is unavailable, the value `Not available` is passed.

### Zero-shot prompt template

```text
SYSTEM:
You explain Python notebook dependency errors to researchers who may not be programmers.

Write in plain, everyday language. Avoid technical jargon. If a technical term is necessary, explain it briefly.

Use only the information provided. Do not invent package versions, installation commands, file contents, or causes that are not supported by the error.

Write 2–4 short sentences. Do not use headings, bullet points, or JSON.
Do NOT suggest a fix or a command. Only explain what went wrong and why the notebook stopped.

Rules:
- For ModuleNotFoundError, describe the problem as a missing or unavailable dependency.
- Do not describe a ModuleNotFoundError as a version or compatibility problem unless the error explicitly says so.
- For an ImportError stating that a name cannot be imported from an installed package, explicitly say that the notebook likely expects a different version of that package.
- Do not claim that something was installed incorrectly unless that information is provided.
- Do not use placeholders in the response.
- Do not use metaphors or analogies.

USER:
A Jupyter notebook failed because of a dependency-related error.

Error type:
{error_type}

Module or import involved:
{failing_module}

Error message or traceback:
{traceback}

Failing code cell:
{cell_source}

Explain, in plain language:
1. what the error means;
2. which module or import is involved;
3. why the notebook could not continue;
4. whether the available information indicates a missing dependency or a likely compatibility/version problem.

Return only the explanation text.
```

### Expected output style

For:

```text
Error type: ModuleNotFoundError
Module or import involved: sklearn
Error message: No module named 'sklearn'
```

an appropriate explanation is:

```text
The notebook stopped because Python could not find the `sklearn` module. This means the library that provides this module is not available in the environment where the notebook ran. Because the import failed, the notebook could not continue with that code cell.
```

### Few-shot variant

The `few_shot` strategy uses the same template but adds two or three example error–explanation pairs before the real input.

The examples demonstrate concise, plain-language explanations without commands or repair recommendations. The concrete examples are added later in this document after the repair prompt design is complete.

### Design rationale

The prompt supports NFR2 by requiring plain-language, non-expert framing.

It keeps `LLMExplainer` separate from `RAGRepairAgent`, consistent with the component contracts: the explainer produces explanation text, while the repair agent produces the structured fix object.

The template is config-injected through named slots, supporting NFR5.

### i3 Deviation: Structured JSON Output Instead of Prose

L4's zero-shot template above was designed for **plain-prose output**: "Write 2–4 short sentences. Do not use headings, bullet points, or JSON." That framing predates the i2 context-extraction output and the i3 implementation work.

The i3 implementation (`prompts/dependency_explanation_v1.txt`) deviates from this and requires `LLMExplainer` to return **schema-valid JSON** instead, validated against `schemas/explanation.schema.json`:

```json
{
  "summary": "...",
  "root_cause": "...",
  "evidence": ["..."],
  "failing_module": "... or null",
  "explanation_confidence": "high | medium | low",
  "limitations": "..."
}
```

Reasons for the deviation:

- i3's acceptance criteria require the explainer to "produce schema-valid JSON explanations" and log structured results to `data/llm-explanations/explanation_results.jsonl`, which needs machine-readable output rather than free text.
- L4's own JSON validity strategy (§9) already established a validated pattern - schema-constrained output, local parsing/validation, retry-on-failure - for the repair prompt. i3 extends that same pattern to the explainer instead of introducing a second, inconsistent output-handling approach.
- The explainer's "no repair suggestion" boundary from L4 is preserved under JSON just as it was under prose: the schema has no `fix`/`command`/`version` field, and `additionalProperties: false` prevents one from being smuggled in.

The output field `explanation_confidence` is deliberately named to avoid colliding with i2's own `confidence` field (the classifier's confidence in the subtype/root-cause hint, computed in i2 and passed into the prompt as `classifier_confidence`). The two are conceptually different - one is the upstream classifier's confidence, the other is the LLM's self-assessed confidence in its own explanation - and are expected to diverge (e.g. high classifier confidence in a pattern match, but only medium explanation confidence when the exact root cause can't be fully verified from the available context).

This is a design deviation from L4, not a correction of it: L4's prose-only design remains valid documentation of the original prompt-design intent; i3 supersedes it for the actual implementation.

## 8. Repair Prompt

### Purpose

The repair prompt is used by `RAGRepairAgent`.

Its purpose is to convert a dependency-related notebook failure into the existing §6.1 structured fix object. It is repair-only and does not produce a user-facing explanation.

The output is parsed and validated against the §6.1 fix object. Its fields are then mapped to the `repair_attempts` table by the ResultLogger.

### Input slots

The prompt receives:

- `error_context`
- `current_requirements`
- `notebook_imports`
- `pypi_versions`
- `prior_attempt` — optional; used only in the second repair round

`error_context` contains the classified error details, including the error type, failing module, and traceback.

If a context field is unavailable, the value `Not available` is passed.

### Repair rules

- Allowed actions are only `install`, `pin_version`, and `none`.
- Use `install` only for a missing, pip-installable dependency.
- Use `pin_version` only when the available PyPI data supports a specific compatible version.
- A `pin_version` result must include a non-null `version` and a matching exact install command.
- Use `none` when no safe pip-only fix can be proposed, when the issue requires a source-code or environment-path change, or when version evidence is insufficient.
- Do not suggest `apt`, `conda`, `pip uninstall`, notebook code edits, import replacement, or environment-variable changes.
- The import name and pip install name may differ. Both fields must be returned.
- Do not assume that an import name is also a pip package name. If no verified mapping is available, use `none`.

### Zero-shot prompt template

```text
SYSTEM:
You generate safe, structured pip-based repair proposals for Python dependency errors in Jupyter notebooks.

Return ONLY one valid JSON object. Do not include markdown, explanations outside JSON, or additional text.

Allowed actions:
- "install": a third-party dependency is missing and can be installed with pip.
- "pin_version": a package version is likely incompatible, and the provided PyPI data supports one specific version.
- "none": no safe pip-only fix can be proposed.

Rules:
- Use only the supplied information.
- Do not invent package names, package versions, Python-version compatibility, or PyPI evidence.
- For "install", set "version" to null.
- For "install", "install_name" and "command" must be non-null. The command must be exactly `pip install <install_name>`.
- For "pin_version", "install_name", "version", and "command" must be non-null. The command must be exactly `pip install <install_name>==<version>`.
- For "pin_version", "version" must be a non-null version supported by the provided PyPI data.
- If a safe version cannot be supported by the supplied PyPI data, use "none".
- For "none", set "install_name", "version", and "command" to null.
- Do not propose apt, conda, pip uninstall, source-code edits, import replacements, or path changes.
- The import name and pip install name may differ. Return both names.
- Known examples include: sklearn → scikit-learn; umap → umap-learn; pkg_resources → setuptools.
- Do not assume that an import name is also a pip package name. For an unknown import name, use "none" unless a verified import-to-package mapping is provided in the input or listed in the known examples.
- When using "none" because the pip package name is unknown, state in the rationale that the pip package name cannot be safely determined from the available information.

Return exactly this JSON structure:
{
  "action": "install | pin_version | none",
  "import_name": "string",
  "install_name": "string or null",
  "version": "string or null",
  "command": "string or null",
  "rationale": "one short sentence",
  "pypi_evidence": {
    "latest_version": "string or null",
    "chosen_version": "string or null",
    "requires_python": "string or null"
  }
}

USER:
Dependency error context:
{error_context}

Current requirements file:
{current_requirements}

Notebook imports:
{notebook_imports}

Available PyPI version data:
{pypi_versions}

Prior repair attempt, if any:
{prior_attempt}

Suggest the safest valid pip-only repair.
```

### Expected output style

For a missing `sklearn` import:

```json
{
  "action": "install",
  "import_name": "sklearn",
  "install_name": "scikit-learn",
  "version": null,
  "command": "pip install scikit-learn",
  "rationale": "The sklearn import is missing and is provided by the scikit-learn package.",
  "pypi_evidence": {
    "latest_version": null,
    "chosen_version": null,
    "requires_python": null
  }
}
```

### Version-mismatch handling before L5

Before L5 supplies PyPI version data, the prompt must not guess a pinned version.

For a likely version mismatch without grounded PyPI data, a safe output is:

```json
{
  "action": "none",
  "import_name": "scipy",
  "install_name": null,
  "version": null,
  "command": null,
  "rationale": "A specific compatible scipy version cannot be selected safely without PyPI version evidence.",
  "pypi_evidence": {
    "latest_version": null,
    "chosen_version": null,
    "requires_python": null
  }
}
```

### Design rationale

The prompt reuses the §6.1 fix object rather than defining a new repair format. It limits the v1 repair scope to `install`, `pin_version`, and `none`.

The RAG slot provides live PyPI metadata so version choices are grounded rather than hallucinated. The optional `prior_attempt` slot supports the later two-round repair ablation without changing the prompt structure.

## 9. JSON Validity Strategy

### Decision

The repair prompt requires machine-readable JSON because its output is consumed by later pipeline components.

The selected strategy is:

1. use Ollama structured output with the §6.1 fix-object JSON schema;
2. parse and validate the returned JSON locally;
3. retry once with a stricter correction prompt if parsing or validation fails;
4. log the failed attempt if the retry also fails, without applying a command.

### Primary output control

The primary repair request uses Ollama schema-constrained JSON output. The schema corresponds to the existing §6.1 fix object and does not define a separate repair format.

The prompt also repeats the expected JSON structure. This gives the model both runtime-level output constraints and clear task-level instructions.

### Local validation

After receiving the model response, the pipeline must:

1. parse the response as JSON;
2. verify that all required fields exist;
3. verify that `action` is one of:
   - `install`
   - `pin_version`
   - `none`

4. verify action-dependent rules:
   - `install` requires an `install_name` and command, while `version` is `null`;
   - `pin_version` requires an `install_name`, a non-null `version`, and a matching command;
   - `none` must not propose a command;

5. reject outputs containing unsupported actions or non-pip commands.

### Malformed-output retry

If the first model output is not valid JSON or fails schema validation, the system retries once with a correction prompt.

The retry prompt provides the invalid output and instructs the model to return only one valid JSON object matching the §6.1 fix object. It must not alter the underlying error context.

If the second response also fails validation, the repair attempt is logged as an LLM-output failure. No command is passed to the FixApplicator.

### Rationale

Schema-constrained output reduces malformed JSON. Local validation and one retry ensure that invalid or unsafe output does not stop the batch pipeline and does not result in an unsafe repair command.

This design follows the architecture requirement that malformed LLM output is logged while one failed notebook does not crash the overall batch.

## 10. Few-Shot Examples

The `few_shot` strategy uses the same prompt templates as `zero_shot`, but adds solved examples before the real notebook input.

The examples demonstrate the expected explanation style, JSON structure, and import-name to install-name mapping.

### 10.1 Explanation Few-Shot Examples

Insert the following examples before the real error input when `prompt_strategy = few_shot`:

```text
--- Example 1 ---
Error type: ModuleNotFoundError
Import involved: sklearn
Error message: No module named 'sklearn'
Explanation: The notebook stopped because Python could not find the `sklearn` module. The library that provides it is not available in the environment where the notebook ran. Because the import failed, that code cell could not continue.

--- Example 2 ---
Error type: ImportError
Import involved: scipy
Error message: cannot import name 'cumtrapz' from 'scipy.integrate'
Explanation: The notebook found the scipy library, but it could not find the `cumtrapz` feature that the notebook expected. This suggests a compatibility problem between the notebook and the installed scipy version. Because the import failed, the notebook could not continue running.

--- End examples ---
```

### 10.2 Repair Few-Shot Examples

Insert the following examples before the real error input when `prompt_strategy = few_shot`:

```text
--- Example 1 ---
Import involved: sklearn
Error: No module named 'sklearn'
PyPI data: Not available
Output:
{
  "action": "install",
  "import_name": "sklearn",
  "install_name": "scikit-learn",
  "version": null,
  "command": "pip install scikit-learn",
  "rationale": "The sklearn import is missing and is provided by the scikit-learn package.",
  "pypi_evidence": {
    "latest_version": null,
    "chosen_version": null,
    "requires_python": null
  }
}

--- Example 2 ---
Import involved: umap
Error: No module named 'umap'
PyPI data: Not available
Output:
{
  "action": "install",
  "import_name": "umap",
  "install_name": "umap-learn",
  "version": null,
  "command": "pip install umap-learn",
  "rationale": "The umap import is missing and is provided by the umap-learn package.",
  "pypi_evidence": {
    "latest_version": null,
    "chosen_version": null,
    "requires_python": null
  }
}

--- End examples ---
```

### 10.3 Prompt Strategy Use

The two supported L4 prompt strategies are:

- `zero_shot`: uses the base prompt template without examples;
- `few_shot`: uses the same base template with the relevant examples inserted before the real input.

The examples are based on real development-sample cases. They include both a missing-package case and an import-name to install-name mismatch.

## 11. Prompt Strategies

The selected L4 prompt strategies are recorded in the `prompt_strategy` field of each later `repair_attempts` row.

### Selected strategies

| Value       | Explanation prompt                                                   | Repair prompt                                                   |
| ----------- | -------------------------------------------------------------------- | --------------------------------------------------------------- |
| `zero_shot` | Uses the base explanation template without examples                  | Uses the base repair template without examples                  |
| `few_shot`  | Uses the base explanation template plus the examples in Section 10.1 | Uses the base repair template plus the examples in Section 10.2 |

### Decision

The initial L4 comparison uses only:

```text
zero_shot
few_shot
```

Chain-of-thought prompting is not included as a separate L4 strategy.

The selected models are locally deployable 7B–13B-class models, for which the benefit of additional reasoning prompts is uncertain. Adding chain-of-thought would also introduce another prompt technique and make the comparison less controlled.

### Consistent use per run

A single run must use the same prompt strategy for both components:

```text
LLMExplainer: zero_shot or few_shot
RAGRepairAgent: zero_shot or few_shot
```

For `few_shot`, examples are inserted after the system instructions and before the real notebook error.

When a development-sample row is also used as a few-shot example, that same row must not be tested as the real input in that run. This prevents the model from simply copying the demonstrated answer.

### Default strategy

The initial default for L4 sanity checks is:

```text
few_shot
```

Few-shot prompting is expected to improve consistency of explanation style, JSON structure, and import-name to install-name mapping. The later evaluation will compare it against `zero_shot`.



## 12. Design-Time Sanity Check

Prompt templates were manually tested with the local model `gemma2:9b` through Ollama using dependency-error examples from `data/prompt-tests/dev_sample.csv`.

Eight prompt/output examples were saved under `data/prompt-tests/`:

- `example_01_explanation_sklearn.md`
- `example_02_explanation_scipy.md`
- `example_03_repair_sklearn.md`
- `example_04_repair_umap.md`
- `example_05_repair_scipy_no_evidence.md`
- `example_06_few_shot_explanation_pkg_resources.md`
- `example_07_few_shot_repair_dms_variants.md`
- `example_08_repair_scipy_with_evidence.md`

`example_08_repair_scipy_with_evidence.md` is a controlled `pin_version` branch test. It reuses the SciPy case with supplied compatibility evidence and is not part of the later zero-shot versus few-shot strategy comparison.

The tests confirmed that the explanation prompt can distinguish between missing dependencies and likely version incompatibilities without suggesting repairs.

The repair prompt produced valid JSON for all final test cases. It correctly handled:
- known import-to-package mappings such as `sklearn` → `scikit-learn` and `umap` → `umap-learn`;
- safe abstention when a package name or compatible version could not be verified;
- version pinning when supplied compatibility evidence supported a specific version.

During testing, additional safety rules were added to prevent unsupported package-name mappings, missing install commands, and unsupported version claims.

`few_shot` is selected provisionally as the default prompt strategy because the tested few-shot prompts followed the required output format, produced a clear explanation, and handled safe abstention correctly. This design-time check does not establish that few-shot prompting produces higher repair success than zero-shot prompting. `zero_shot` remains available as the baseline strategy.

This was a design-time sanity check only. It does not measure repair success after notebook execution. Runtime repair evaluation will be performed later through the FixApplicator and pipeline reruns.
## I3 implementation update: LLMExplainer prompt execution

The i3 implementation changes the LLMExplainer from the original L4 plain-text explanation format to schema-valid JSON output. This is necessary because explanations must be automatically validated, retried on malformed output, logged consistently, and evaluated later.

Final i3 output schema:

- `summary`
- `root_cause`
- `evidence`
- `failing_module`
- `explanation_confidence`
- `limitations`

The schema is defined in:

`schemas/explanation.schema.json`

The prompt template is:

`prompts/dependency_explanation_v1.txt`

The runtime configuration is:

`config/llm_explainer.yaml`

### Runtime and retry behavior

The i3 runner uses Ollama with `gemma2:9b` as the local explanation model. The runner supports config-gated retry behavior for:

- invalid JSON
- schema validation errors
- timeout
- model unavailable

The retry policy is controlled by `retry.max_retries` and `retry.retry_on` in `config/llm_explainer.yaml`.

### Fallback-model decision

For i3, the LLMExplainer does not switch to a different fallback model. This follows the L4 component decision that the explainer uses Gemma-2 9B. Fallback-model switching is deferred to the later repair component, where CodeLlama or DeepSeek-Coder may be useful for repair-command generation.

Therefore, `models.fallback: null` is intentional for `LLMExplainer`.

### Logged result structure

Explanation results are written as JSONL to:

`data/llm-explanations/explanation_results.jsonl`

Each row uses the nested structure:

- `input`: original i2 metadata and classification fields
- `llm`: model and prompt metadata
- `explanation_result`: generated explanation, validation status, latency, token counts, retry attempts, and errors

This separation prevents raw input metadata, such as original error messages, from being confused with generated explanation content.

### Scope boundary

The LLMExplainer only explains dependency-related failures. It does not generate, apply, or validate fixes. Repair generation remains assigned to the later RAGRepairAgent/FixApplicator components.

### i4 scope clarification: explanation covers all 214 rows, repair covers 200

Made explicit while implementing i4 (`RAGRepairAgent`), because that is the first component whose
correctness depends on the answer.

`LLMExplainer` processes every `DEPENDENCY_ERROR` row - all 214, across all four subtypes
(`missing_package`, `wrong_version`, `system_library`, `mapping_unknown`) - matching the Vision
Doc's O1 definition, which is scoped to "a dependency-related error message" with no
pip-fixability qualifier. The "v1 repair scope" language throughout L4 (§2, §6) and in
`docs/dataset.md` was always about repair eligibility, not explanation eligibility; it is
restated here because earlier drafts of `docs/architecture-note.md` and `thesis/architecture.tex`
gated `LLMExplainer` behind the same "in scope" signal as `RAGRepairAgent`, which read as a
stricter shared gate than the objective actually states.

Concretely:

- `RAGRepairAgent` and `FixApplicator` (O2/O3) operate only on rows where `scope_status ==
  "usable"` (200 of 214) - unchanged from the original pip-only v1 scope.
- `LLMExplainer` (O1) has no such filter, in the prompt template or in the runner. The i3
  acceptance criteria never restricted it to pip-fixable rows, and the shipped prompt template
  (`prompts/dependency_explanation_v1.txt`) already lists `system_level_dependency` and
  `local_import_or_path_issue` - the root-cause hints for the two excluded subtypes - as known
  values, confirming this was the template's intent even before the scope conflict was written
  down explicitly.
- `scope_status`, `exclusion_reason`, and `split` are threaded through i2's enriched JSONL and
  into i3's logged `input` block purely for traceability and later filtering - the explanation
  prompt itself is not given a reason to treat excluded subtypes differently, since it must not
  claim repairability for any subtype regardless (`schemas/explanation.schema.json` has no
  fix/command/version field).
- The 13 `dev`-split rows (used to design and sanity-check both prompts in L4) must not be scored
  in a final explanation-quality evaluation, separately from the pip-scope question above.
