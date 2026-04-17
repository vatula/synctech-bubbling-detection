### Redaction Ledger

### Feature ID
- `F5-RPT-001`: Enhance `phase5_consolidated_report` observability for localization diagnostics.

### Gap Analysis (Completed Before Redaction Authoring)
- Gap 1: `results/phase5_consolidated_report.{json,md}` currently omits exact hardware identity used during evaluation.
  - Root cause: `src/models/evaluation.py` report schema has no runtime/hardware block.
- Gap 2: Report omits input image resolution and effective inference batch size.
  - Root cause: data transforms and evaluation loops use fixed/implicit values, but these values are not surfaced into report payload.
- Gap 3: Report omits explicit anomaly-localization architecture identity (e.g., Dinomaly/PatchCore/PaDiM + backbone details).
  - Root cause: localization model metadata is used internally but not persisted in report schema.
- Impact: AUROC (`0.703297`) and localization latency (`2019.002488 ms`) cannot be root-caused reliably without runtime/model provenance.

### Micro-Experiments (Executed)
- Experiment A (`uv run python -c ...torch.cuda...`): verified hardware metadata is programmatically available.
  - Observed: `cuda_available=true`, `device_count=1`, `gpu_model="AMD Radeon Graphics"`, `torch_version="2.11.0+rocm7.2"`.
- Experiment B (`uv run python -c ..._load_samples...yaml...`): verified report context metadata is programmatically extractable.
  - Observed: sample tensor shape `[3, 224, 224]` (input resolution `224x224`), `sample_count=34`.
  - Observed: effective evaluation path batch size is `1` for classification/localization loops.
  - Observed: localization architecture is discoverable from config as `anomalib.models.Dinomaly`; encoder provenance is `dinov2reg_vit_large_14`.
- Conclusion: adding runtime/model provenance fields is low-risk and materially improves interpretability of localization quality/latency.

### Exact Planned Redactions

#### 1) Source Code Redactions
- File: `src/models/evaluation.py`
  - Add `RuntimeContext` typed block to consolidated schema with exact fields:
    - `device_type`, `cuda_available`, `gpu_model`, `torch_version`, `rocm_hip_version`.
  - Add `InferenceContext` typed block with exact fields:
    - `input_resolution_hw`, `classification_batch_size`, `localization_batch_size`, `semantic_batch_size`.
  - Add `LocalizationContext` typed block with exact fields:
    - `architecture`, `encoder_name`, `checkpoint_path`, `score_threshold`.
  - Add helper collectors:
    - `_collect_runtime_context()`
    - `_collect_inference_context(samples)`
    - `_collect_localization_context(checkpoint_path)`
  - Extend `ConsolidatedReport` payload and `_render_markdown()` to emit all new context sections.
  - Ensure values are deterministic and serializable in JSON/Markdown outputs.

- File: `src/models/localization.py`
  - Add metadata accessor (or equivalent safe property exposure) for architecture/backbone provenance, avoiding duplicate checkpoint parsing in downstream reporters.
  - Preserve current inference behavior; no threshold/logic regressions.

- File: `tests/test_phase5_report_context.py` (new)
  - Add assertions that generated consolidated report contains:
    - non-empty hardware identity,
    - exact input resolution tuple,
    - explicit inference batch sizes,
    - explicit localization architecture string.
  - Add negative assertion to fail when architecture/runtime blocks are missing.

#### 2) PLAN.md Redactions
- File: `PLAN.md`
  - Redact `Phase 5 -> Task 5.2` text to explicitly require runtime provenance in consolidated report:
    - exact hardware/GPU model,
    - input image resolution,
    - inference batch size,
    - localization architecture/backbone identity.
  - Add acceptance clause: report is incomplete unless provenance block is present in both `.json` and `.md` outputs.

#### 3) AGENTS.md Redactions
- File: `AGENTS.md`
  - Redact `Phase 5 -> Task 5.2` checklist item to include mandatory context logging requirements:
    - hardware profile,
    - inference tensor resolution,
    - effective inference batch size,
    - localization model architecture provenance.
  - Add verification note under Phase 5 execution discipline:
    - reject Phase 5 evaluation output if provenance fields are missing.

### Sequencing for Implementation (Next Execution Unit)
1. Implement report schema/context collectors in `src/models/evaluation.py`.
2. Add/adjust localization metadata exposure.
3. Generate tests for provenance enforcement.
4. Execute static gates and tests.
5. Update `PLAN.md` and `AGENTS.md` according to redactions above.

### Verification Commands for the Planned Edits
- `uv run ruff check --fix .`
- `uv run ruff format .`
- `uv run pyright`
- `uv run pytest -q tests/test_phase5_report_context.py`
- `uv run python -m src.models.evaluation`

### Definition of Done for `F5-RPT-001`
- Consolidated report includes runtime, inference, and localization architecture provenance in both JSON and Markdown outputs.
- Report generation remains backward-stable for existing metric fields.
- Static analysis passes with zero errors (`ruff`, `pyright`).
- New provenance tests pass.