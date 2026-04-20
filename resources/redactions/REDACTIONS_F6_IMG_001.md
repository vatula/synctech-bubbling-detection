### Redaction Ledger

### Feature ID
- `F6-IMG-001`: Centralize configurable input image dimension and remove repeated hard-coded `224` defaults.

### Gap Analysis (Completed Before Redaction Authoring)
- Gap 1: Image-size default is duplicated in runtime transform paths.
  - Evidence: `src/data/transforms.py` previously used `image_size: int = 224` in both `get_train_transforms(...)` and `get_inference_transforms(...)`.
  - Impact: changing input size requires multiple edits and increases drift risk.

- Gap 2: Distillation runtime duplicates image-size defaults and relies on implicit fixed student-resolution assumptions.
  - Evidence: `src/models/distillation.py` previously used `image_size: int = 224` in distillation dataloader construction.
  - Impact: inconsistent image-size propagation between data transforms and model-building paths.

- Gap 3: Tests and report-context fixtures repeated fixed `224` literals.
  - Evidence: `tests/test_augmentations.py`, `tests/test_phase4_semantics.py`, and `tests/test_phase5_report_context.py` used direct `224` shape/resolution assertions.
  - Impact: weak regression signal for single-source configuration and unnecessary literal duplication.

- Gap 4: No centralized input-size bounds policy existed to enforce accepted runtime range.
  - Required constraint: image input must be `>=224` and `<=512`.
  - Impact: invalid sizes could be accepted silently if introduced through refactors or runtime overrides.

### Micro-Experiments (Executed Before/Alongside Redaction)
- Experiment A (`search_project "224" src`): cataloged literal usages and separated true input-size defaults from unrelated normalization values (`0.224`).
- Experiment B (targeted module scan): verified transform/dataloader paths are primary resolution entry points and highest-leverage refactor targets.
- Experiment C (test fixture scan): confirmed assertion-level hard-coded dimension literals that should align with centralized defaults.

### Exact Planned Redactions

#### 1) Source Code Redactions
- File: `src/utils/image_size.py` (new)
  - Introduce single-source constants and policy:
    - `MIN_IMAGE_SIZE = 224`
    - `MAX_IMAGE_SIZE = 512`
    - `DEFAULT_IMAGE_SIZE = 224`
    - `IMAGE_SIZE_ENV_KEY = "BUBBLING_IMAGE_SIZE"`
  - Add validated resolver API:
    - `validate_image_size(...)`
    - `get_default_image_size(...)`
    - `resolve_image_size(...)`

- File: `src/data/transforms.py`
  - Replace duplicated hard-coded defaults with `resolve_image_size(...)`.
  - Keep behavior identical at default runtime (`224`) while enabling one-place override policy.

- File: `src/models/distillation.py`
  - Replace distillation dataloader hard-coded default with centralized resolver.
  - Route student-construction image size through centralized policy (bounded and override-capable).

- File: `src/models/extractor.py`
  - Redact fixed-shape docstring wording from `(B, 3, 224, 224)` to `(B, 3, H, W)`.

#### 2) Test Redactions
- File: `tests/test_image_size.py` (new)
  - Add coverage for:
    - default-in-bounds behavior,
    - min/max acceptance,
    - out-of-range rejection,
    - env override parsing and validation.

- Files: `tests/test_augmentations.py`, `tests/test_phase4_semantics.py`, `tests/test_phase5_report_context.py`
  - Replace repeated `224` literals with `DEFAULT_IMAGE_SIZE` from centralized config.
  - Preserve original semantic assertions.

#### 3) Documentation/Reporting Redactions
- File: `results/phase5_consolidated_report.md` (context reference)
  - Keep existing historical value as recorded evidence.
  - Ensure forward generation paths consume centralized runtime resolution instead of scattered literals.

### Sequencing for Implementation (Execution Unit)
1. Add centralized image-size policy module and bound checks.
2. Refactor transform and distillation entry paths to consume centralized policy.
3. Update tests + add dedicated bounds/regression tests.
4. Run static analysis and relevant tests.
5. Confirm no remaining refactorable hard-coded input-size literals in active runtime paths.

### Verification Commands for Planned Edits
- `uv run ruff check --fix .`
- `uv run ruff format .`
- `uv run pyright`
- `uv run pytest -q tests/test_image_size.py tests/test_augmentations.py tests/test_phase4_semantics.py tests/test_phase5_report_context.py`

### Definition of Done for `F6-IMG-001`
- Input image dimension is configurable from a single policy source and enforced within `[224, 512]`.
- Runtime transform and distillation paths no longer duplicate hard-coded `224` defaults.
- Relevant tests validate bounds and centralized-default usage.
- Static analysis passes with zero errors (`ruff`, `pyright`).