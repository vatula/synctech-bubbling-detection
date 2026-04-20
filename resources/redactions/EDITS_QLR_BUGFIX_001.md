# EDITS_QLR.md

## Mission
Wire existing QLoRA configuration into the distillation execution path so that QLoRA is not only defined in `src/models/qlora_config.py` but is operationally consumed by `src/models/distillation.py` and `src/models/retrain.py`.

## Hard Constraints (Do Not Violate)
- Execute exactly one task at a time in the order listed below.
- Do not edit files not explicitly listed in each task.
- No `print()` calls anywhere. Use existing `structlog` logger patterns.
- Preserve current default behavior unless explicitly changed by this checklist.
- After each task, run required gates and stop.
- If any task cannot be completed exactly as specified, stop and report `BLOCKED` with exact reason.

## Global Quality Gates (Run After Every Task)
- `uv run ruff check --fix .`
- `uv run ruff format .`
- `uv run pyright`

Pass condition per task: zero errors from all three commands.

## Definition of Done
- QLoRA settings are loaded from file and applied to Qwen teacher model through PEFT during distillation setup.
- Distillation CLI and retrain CLI can control QLoRA enablement and trainability.
- When QLoRA trainability is enabled, adapter parameters are included in optimization.
- Tests verify wiring, argument parsing, and checkpoint metadata.
- Existing classifier/localization/semantic flows remain backward compatible.

---

## Task QLR-00: Preflight Baseline
Status: [ ]

Objective: establish a known baseline before touching code.

Allowed files: none (read/execute only).

Required actions:
- Run baseline tests:
- `uv run pytest tests/test_phase4_semantics.py tests/test_distillation_epoch_config.py -q`
- Capture current pass/fail summary.

Validation gate:
- Run Global Quality Gates.
- Stop.

---

## Task QLR-01: Add QLoRA Config Loader API
Status: [ ]

Objective: make `src/models/qlora_config.py` usable by runtime code, not only by notebook utilities.

Allowed files:
- `src/models/qlora_config.py`
- `tests/test_phase4_semantics.py`

Required code changes:
- In `src/models/qlora_config.py`, add:
- `load_qlora_config(path: Path) -> QLoRASettings`
- `qlora_settings_from_payload(payload: dict[str, Any]) -> QLoRASettings`
- Strict validation:
- required keys: `task_type`, `r`, `lora_alpha`, `lora_dropout`, `bias`, `target_modules`
- `target_modules` must include all: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `r` and `lora_alpha` must be positive ints
- `lora_dropout` in `[0.0, 1.0)`
- Add structlog telemetry for successful load and validation failures.
- Keep existing public functions unchanged (`build_lora_config`, `qlora_config_payload`, `save_qlora_config`).

Required tests:
- Extend `tests/test_phase4_semantics.py` with unit tests for:
- valid payload -> `QLoRASettings`
- missing keys -> `ValueError`
- invalid `target_modules` -> `ValueError`

Validation gate:
- `uv run pytest tests/test_phase4_semantics.py -q`
- Run Global Quality Gates.
- Stop.

---

## Task QLR-02: Distillation CLI Wiring for QLoRA
Status: [ ]

Objective: expose explicit runtime controls for teacher-side QLoRA from CLI/env.

Allowed files:
- `src/models/distillation.py`
- `tests/test_distillation_epoch_config.py`

Required code changes:
- Add env keys with defaults in `distillation.py`:
- `DISTILLATION_TEACHER_LORA_ENABLED` default `false`
- `DISTILLATION_TEACHER_LORA_CONFIG` default `results/phase4/task_4_2_qlora_config.json`
- `DISTILLATION_TEACHER_LORA_TRAINABLE` default `false`
- `DISTILLATION_TEACHER_LORA_LR` default `5e-5`
- Add CLI args:
- `--teacher-lora-enabled` choices `true|false`
- `--teacher-lora-config` path
- `--teacher-lora-trainable` choices `true|false`
- `--teacher-lora-lr` float > 0
- Parse booleans robustly and validate LR > 0.
- Preserve existing args and defaults.

Required tests:
- Extend `tests/test_distillation_epoch_config.py`:
- CLI override for each new arg
- env fallback for each new arg
- invalid LR rejected

Validation gate:
- `uv run pytest tests/test_distillation_epoch_config.py -q`
- Run Global Quality Gates.
- Stop.

---

## Task QLR-03: Apply QLoRA to Teacher Model
Status: [ ]

Objective: attach LoRA adapters to the Qwen teacher model at runtime when enabled.

Allowed files:
- `src/models/distillation.py`
- `tests/test_phase4_semantics.py`

Required code changes:
- Import and use `load_qlora_config` and `build_lora_config` from `src.models.qlora_config`.
- Import PEFT adapter application function (`get_peft_model`).
- In `QwenTeacherEncoder.__init__`, add parameters:
- `teacher_lora_enabled: bool = False`
- `teacher_lora_config_path: Path | None = None`
- `teacher_lora_trainable: bool = False`
- `teacher_lora_lr: float = 5e-5`
- Behavior when enabled:
- Load QLoRA settings from config path.
- Build `LoraConfig`.
- Wrap teacher model via PEFT.
- Freeze non-LoRA parameters.
- If `teacher_lora_trainable` is false, set LoRA params `requires_grad=False`.
- If `teacher_lora_trainable` is true, set LoRA params `requires_grad=True`.
- Add logging fields:
- `teacher_lora_enabled`
- `teacher_lora_trainable`
- `teacher_lora_config_path`
- `teacher_total_params`
- `teacher_trainable_params`
- Add helper method on teacher encoder:
- `trainable_parameters() -> list[torch.nn.Parameter]`

Required tests:
- Add test that when QLoRA enabled, PEFT wrapper is applied (use monkeypatch/spies; do not require downloading real model).
- Add test that trainable parameter count differs between `teacher_lora_trainable=true` and `false`.

Validation gate:
- `uv run pytest tests/test_phase4_semantics.py -q`
- Run Global Quality Gates.
- Stop.

---

## Task QLR-04: Trainer Optimization Wiring
Status: [ ]

Objective: include teacher LoRA params in optimization only when explicitly trainable.

Allowed files:
- `src/models/distillation.py`
- `tests/test_phase4_semantics.py`

Required code changes:
- In `ContrastiveDistillationTrainer.__init__`:
- Keep student optimizer behavior as baseline.
- Add optional teacher optimizer group when teacher exposes non-empty `trainable_parameters()`.
- Use `teacher_lora_lr` from teacher config for teacher param group.
- In `train_step`:
- If teacher has trainable params, remove teacher `torch.no_grad()` path.
- If teacher has no trainable params, keep current no-grad path.
- Ensure backward/step touches both student and teacher-trainable params when applicable.
- Ensure no shape/behavior regression in contrastive loss.

Required tests:
- Add deterministic unit test with dummy teacher+student:
- no teacher trainable params -> one optimizer group path
- teacher trainable params present -> teacher params receive gradients and step

Validation gate:
- `uv run pytest tests/test_phase4_semantics.py -q`
- Run Global Quality Gates.
- Stop.

---

## Task QLR-05: Main Distillation Entry Point Integration
Status: [ ]

Objective: thread new CLI options into actual distillation execution flow.

Allowed files:
- `src/models/distillation.py`

Required code changes:
- In `main()`:
- instantiate `QwenTeacherEncoder` with all new CLI-derived QLoRA args.
- Preserve warmup embedding shape logic.
- Ensure existing student architecture selection behavior remains unchanged.
- Add a startup log block summarizing active QLoRA mode and trainability.

Validation gate:
- `uv run pytest tests/test_distillation_epoch_config.py tests/test_phase4_semantics.py -q`
- Run Global Quality Gates.
- Stop.

---

## Task QLR-06: Retrain Pipeline Arg Propagation
Status: [ ]

Objective: allow orchestration layer to control QLoRA distillation mode.

Allowed files:
- `src/models/retrain.py`
- `tests/test_distillation_epoch_config.py`

Required code changes:
- Extend `build_distillation_command(...)` to include:
- `teacher_lora_enabled`
- `teacher_lora_config`
- `teacher_lora_trainable`
- `teacher_lora_lr`
- Extend retrain CLI with matching args and env defaults.
- Ensure SIGSEGV CPU fallback retains QLoRA args.

Required tests:
- Update `tests/test_distillation_epoch_config.py` to assert:
- new args appear in generated distillation command
- fallback command preserves QLoRA args while adding `--teacher-device cpu`

Validation gate:
- `uv run pytest tests/test_distillation_epoch_config.py -q`
- Run Global Quality Gates.
- Stop.

---

## Task QLR-07: Distillation Checkpoint Metadata
Status: [ ]

Objective: persist QLoRA run context for reproducibility.

Allowed files:
- `src/models/distillation.py`
- `tests/test_phase4_semantics.py`

Required code changes:
- Extend `save_distillation_checkpoint(...)` payload with:
- `teacher_lora_enabled: bool`
- `teacher_lora_trainable: bool`
- `teacher_lora_config_path: str | None`
- `teacher_lora_lr: float | None`
- `teacher_lora_target_modules: list[str] | None`
- Maintain backward compatibility for existing keys.

Required tests:
- unit test that checkpoint contains new metadata keys.

Validation gate:
- `uv run pytest tests/test_phase4_semantics.py -q`
- Run Global Quality Gates.
- Stop.

---

## Task QLR-08: End-to-End Focused Validation
Status: [ ]

Objective: verify QLoRA path is operational and baseline path still works.

Allowed files:
- no source edits in this task.

Required actions:
- Baseline (QLoRA off):
- `uv run python -m src.models.distillation --epochs 1 --teacher-lora-enabled false`
- QLoRA enabled but frozen adapters:
- `uv run python -m src.models.distillation --epochs 1 --teacher-lora-enabled true --teacher-lora-trainable false --teacher-lora-config results/phase4/task_4_2_qlora_config.json`
- QLoRA enabled and trainable adapters:
- `uv run python -m src.models.distillation --epochs 1 --teacher-lora-enabled true --teacher-lora-trainable true --teacher-lora-lr 5e-5 --teacher-lora-config results/phase4/task_4_2_qlora_config.json`

Expected results:
- All three commands complete without exceptions.
- Logs show QLoRA mode and trainable parameter counts.
- Distillation checkpoint written successfully each run.

Validation gate:
- Run Global Quality Gates.
- Stop.

---

## Task QLR-09: Regression Test Sweep
Status: [ ]

Objective: ensure no collateral breakage.

Allowed files:
- no source edits in this task.

Required actions:
- `uv run pytest tests/test_phase4_semantics.py tests/test_distillation_epoch_config.py tests/test_phase5_report_context.py tests/test_phase6_requirements.py -q`

Expected results:
- All tests pass.

Validation gate:
- Run Global Quality Gates.
- Stop.

---

## Task QLR-10: Final Integrity Review
Status: [ ]

Objective: verify implementation quality and scope discipline.

Allowed files:
- no source edits in this task.

Required actions:
- Confirm only intended files changed.
- Confirm no `print(` introduced:
- `rg -n "print\(" src tests`
- Confirm QLoRA wiring exists in runtime path:
- `rg -n "teacher-lora|load_qlora_config|get_peft_model|trainable_parameters" src/models/distillation.py src/models/retrain.py`

Expected results:
- No unexpected file changes.
- No print usage.
- Runtime references present.

Validation gate:
- Run Global Quality Gates.
- Stop and request final review.

---

## Non-Negotiable Implementation Notes
- Do not rely on `/notebooks/` for production behavior. Notebook files are gitignored.
- Do not introduce dependency on bitsandbytes unless explicitly requested.
- Keep ONNX/MIGraphX export pipeline unchanged in this checklist.
- Preserve backward compatibility of old checkpoints where possible.

## Deliverable Summary Expected From Implementing Agent
- Short changelog with exact files modified.
- Test command outputs summarized with pass/fail counts.
- Statement confirming QLoRA is now wired into distillation execution path.
