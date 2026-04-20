### Redaction Ledger

### Feature ID
- `F6-LOG-001`: Warning hygiene for full evaluation pipeline (`./evaluate.sh` / `src.models.evaluation`).

### Gap Analysis (Completed Before Redaction Authoring)
- Gap 1: Evaluation starts with raw terminal anomaly: `(null): No such file or directory`.
  - Evidence: first emitted line of `./evaluate.sh` run, before structured logs.
  - Root cause hypothesis: startup/runtime integration issue outside structured logging path (likely dependency/runtime bootstrap), currently untracked and unattributed.
  - Impact: false-negative signal in CI/manual checks; hard to distinguish real file failures from benign startup noise.

- Gap 2: DINOv2 emits fallback warnings due to missing optional acceleration package:
  - `xFormers is not available (SwiGLU)`
  - `xFormers is not available (Attention)`
  - `xFormers is not available (Block)`
  - Root cause: optional fast-path dependency unavailable for current environment; warning emitted from torch hub model internals.
  - Impact: noisy logs + potential lower transformer throughput.

- Gap 3: Lightning emits checkpoint restore state warning:
  - `The dirpath has changed from .../v3/... to .../latest/... therefore ... won't be reloaded`
  - Root cause: checkpoint path aliasing (`latest` vs concrete run directory) across restore/predict flows.
  - Impact: repeated warning noise and ambiguity about exact checkpoint provenance.

- Gap 4: Lightning emits internal deprecation warning:
  - ``isinstance(treespec, LeafSpec)` is deprecated...`
  - Root cause: upstream library compatibility surface (`lightning` + pytree API version).
  - Impact: warning debt; future breakage risk when dependency updates tighten behavior.

- Gap 5: Prediction dataloader performance advisory:
  - `predict_dataloader does not have many workers ... consider num_workers=31`
  - Root cause: inference path uses default dataloader settings with insufficient workers.
  - Impact: avoidable latency overhead during localization evaluation.

- Gap 6: Repeated callback override/reload noise during localization:
  - `callbacks ... will override existing callbacks passed to Trainer`
  - repeated `Restoring states from the checkpoint path ...`
  - Root cause: per-image `Engine.predict(...)` invocation in loop with checkpoint argument; trainer/callback setup repeated N times.
  - Impact: excessive log spam, redundant reload overhead, inflated wall-clock evaluation time.

- Gap 7: Non-actionable cloud-tip noise in local runs:
  - `Tip: ... install litlogger ...`
  - Root cause: default Lightning informational hinting enabled in local evaluation path.
  - Impact: terminal clutter unrelated to project acceptance criteria.

- Overall impact:
  - Full pipeline currently completes, but terminal output is not clean and includes repeated warnings/advisories that obscure real failures and hide low-effort performance wins.

### Micro-Experiments (Required Before Production Redactions)
- Experiment A: isolate startup anomaly origin.
  - Commands:
    - `uv run python -m src.models.evaluation`
    - `uv run python -c "import torch, cv2, anomalib"`
  - Goal: identify whether `(null): No such file or directory` is tied to shell wrapper, import side effects, or runtime loader.

- Experiment B: validate no-reload prediction path.
  - Implement temporary probe in localizer to call `Engine.predict(..., ckpt_path=None)` after model is already loaded.
  - Goal: confirm warning/reload spam disappears while outputs remain stable.

- Experiment C: validate batched localization inference path.
  - Run localization on a small subset (`n=3`) using one predict call and compare scores/masks/boxes to current per-image loop.
  - Goal: preserve functional parity while reducing callback/restore spam.

- Experiment D: confirm worker tuning impact.
  - Compare `num_workers=0` vs tuned worker count for localization inference latency.
  - Goal: remove advisory warning and capture quick latency improvement.

### Exact Planned Redactions

#### 1) Source Code Redactions
- File: `src/models/localization.py`
  - Normalize checkpoint provenance to a resolved canonical path (`Path(...).resolve()`) during localizer initialization.
  - Add a batched inference API (`process_images(...)`) that performs prediction in one engine call and returns per-image outputs.
  - Stop passing `ckpt_path` during each predict call when model weights are already loaded; rely on in-memory model state.
  - Configure inference-time engine/trainer options to reduce non-actionable noise (disable unnecessary logging/progress hooks for evaluation mode, subject to API support).
  - Add explicit `num_workers`/batch controls for predict dataloader wiring.

- File: `src/models/evaluation.py`
  - Replace per-sample localization predict loop with batched localizer invocation.
  - Keep metric semantics unchanged while consuming batched outputs.
  - Add deterministic warning-hygiene bootstrap call at process start.

- File: `src/utils/logger.py`
  - Integrate warning-hygiene setup into `setup_project()` so suppression/rerouting happens before heavy imports/model initialization.

- File: `src/utils/warning_hygiene.py` (new)
  - Add targeted warning filters and one-time structured telemetry for known benign patterns:
    - xFormers missing (fallback expected on current stack).
    - known upstream Lightning deprecation warning (until dependency upgrade lands).
    - optional cloud-tip informational spam.
  - Preserve fail-fast behavior for unknown warnings/errors (do not globally silence warnings).

- File: `pyproject.toml`
  - Add explicit optional dependency group for acceleration/hygiene where feasible (e.g., optional transformer acceleration package when compatible with runtime).
  - Pin or constrain dependency versions to a warning-clean matrix for `anomalib`/`lightning`/`torch` compatibility.

#### 2) Tests Redactions
- File: `tests/test_localization.py`
  - Add/adjust tests to validate canonical checkpoint path handling and batched predict parity against single-image behavior.

- File: `tests/test_phase5_report_context.py`
  - Ensure existing report context assertions remain stable after localization inference refactor.

- File: `tests/test_warning_hygiene.py` (new)
  - Add tests for warning filter registration and one-time structlog telemetry behavior.

- File: `tests/test_evaluation_warning_budget.py` (new)
  - Add integration-style smoke assertion on evaluation output to enforce zero known-warning patterns in terminal logs.

#### 3) Documentation Redactions
- File: `PLAN.md`
  - Redact `Phase 5 -> Task 5.3` acceptance wording to require clean evaluation runtime logs (no unresolved warnings/advisories in local-host validation).
  - Add lightweight warning-budget criterion to `Phase 6 -> Task 6.4` automated assertions.

- File: `AGENTS.md`
  - Add warning-hygiene gate to execution discipline:
    - capture and classify terminal warnings during evaluation,
    - treat repeated dependency/deprecation/configuration warnings as actionable unless explicitly allowlisted.

### Sequencing for Implementation (Next Execution Unit)
1. Run micro-experiments A/B/C/D and capture evidence.
2. Refactor localization prediction path to batched, no-reload inference.
3. Add warning hygiene utility and wire it into startup.
4. Update dependency constraints/options for warning-clean compatibility.
5. Add warning-budget and regression tests.
6. Update `PLAN.md` and `AGENTS.md` with warning-hygiene acceptance gates.
7. Re-run full evaluation and confirm clean terminal output.

### Verification Commands for Planned Edits
- `uv run ruff check --fix .`
- `uv run ruff format .`
- `uv run pyright`
- `uv run pytest -q tests/test_localization.py tests/test_phase5_report_context.py tests/test_warning_hygiene.py tests/test_evaluation_warning_budget.py`
- `./evaluate.sh`

### Definition of Done for `F6-LOG-001`
- Full evaluation pipeline executes to completion with no unresolved warnings/errors/advisories in terminal output.
- Localization metrics/report outputs remain semantically stable after inference-path refactor.
- Warning hygiene is explicit, targeted, and covered by tests (no blanket warning suppression).
- Static analysis passes with zero errors (`ruff`, `pyright`).