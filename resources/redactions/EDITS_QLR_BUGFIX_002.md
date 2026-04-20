# EDITS_QLR_BUGFIX_002.md

## Purpose
This is the authoritative bug-fix execution checklist to achieve **full QLoRA wiring** in distillation.

Current state is partial:
- QLoRA config can be loaded and adapter can be attached.
- Full trainability control, optimizer wiring, provenance metadata, and strict validation are still incomplete.

All goals in this document are mandatory delivery items.

---

## Non-Negotiable Protocol
1. Execute exactly one task at a time, in order.
2. Edit only the files listed in the active task.
3. Update task status brackets in this file at each state transition:
- `[ ]` Not started
- `[~]` In progress
- `[x]` Done (all gates passed)
- `[!]` Blocked
4. On task start: set that task to `[~]` and all later tasks remain `[ ]`.
5. On failure: set that task to `[!]`, write a one-line blocker, stop.
6. Never mark `[x]` before all task gates pass.
7. No `print()` usage in src/tests. Use existing structured logger patterns.
8. Keep behavior backward-compatible unless this plan explicitly changes it.

---

## Mandatory Message Format (for each task)
Use this exact output structure after each task execution:
1. `Executing Task QLR-XXX: <title>`
2. `Checklist Update: QLR-XXX -> [~]`
3. `Files Changed: ...`
4. `Commands Executed: ...`
5. `Result: PASS | BLOCKED`
6. `Checklist Update: QLR-XXX -> [x] | [!]`
7. `Static Analysis: PASSED` (only if all gates pass)
8. `Awaiting authorization for next task.`

---

## Global Gates (run after every task)
```bash
uv run ruff check --fix .
uv run ruff format .
uv run pyright
```

Pass condition: zero errors/warnings in gate output.

---

## Success Criteria (Definition of Done)
All must be true:
- QLoRA settings are strictly validated from JSON before use.
- Distillation CLI has explicit teacher LoRA controls (enabled/config/trainable/lr).
- Retrain CLI and command propagation include the same controls.
- Teacher adapter trainability is controllable and reflected in optimizer behavior.
- Distillation checkpoint stores LoRA provenance metadata.
- Tests verify runtime wiring, argument propagation, validation failures, and metadata persistence.
- Regression tests pass and no unintended scope changes exist.

---

## Task List

### QLR-000: Baseline Snapshot
Status: [x]
Allowlist: none (read/test only)

Actions:
```bash
uv run pytest tests/test_phase4_semantics.py tests/test_distillation_epoch_config.py -q
```
Record summary under a new section `## Baseline Notes` at end of this file.

Task gates:
- Run Global Gates.
- Stop.

---

### QLR-010: Strict QLoRA Payload Validation
Status: [x]
Allowlist:
- `src/models/qlora_config.py`
- `tests/test_phase4_semantics.py`

Required changes:
- Strengthen `qlora_settings_from_payload(payload)`:
  - Required keys exactly: `task_type`, `r`, `lora_alpha`, `lora_dropout`, `bias`, `target_modules`.
  - `r` and `lora_alpha` must be integers and `> 0`.
  - `lora_dropout` must be numeric and in `[0.0, 1.0)`.
  - `bias` must be one of `none|all|lora_only`.
  - `target_modules` must be list/tuple of strings and must include all:
    - `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- Keep existing function names and return types unchanged.
- Add structured logs for successful load and validation error context.

Required tests to add/update:
- valid payload passes.
- missing key fails.
- missing one required target module fails.
- invalid numeric ranges fail.

Task gate:
```bash
uv run pytest tests/test_phase4_semantics.py -q
```
Then Global Gates.

---

### QLR-020: Distillation CLI/Env Controls for Teacher LoRA
Status: [x]
Allowlist:
- `src/models/distillation.py`
- `tests/test_distillation_epoch_config.py`

Required changes:
- Add env-backed defaults:
  - `DISTILLATION_TEACHER_LORA_ENABLED` default `false`
  - `DISTILLATION_TEACHER_LORA_CONFIG` default `results/phase4/task_4_2_qlora_config.json`
  - `DISTILLATION_TEACHER_LORA_TRAINABLE` default `false`
  - `DISTILLATION_TEACHER_LORA_LR` default `5e-5`
- Add CLI args:
  - `--teacher-lora-enabled` (`true|false`)
  - `--teacher-lora-config` (Path)
  - `--teacher-lora-trainable` (`true|false`)
  - `--teacher-lora-lr` (positive float)
- Preserve current args (`--epochs`, `--teacher-device`, `--student-architecture`).
- Keep backward compatibility: old `--qlora-config` alias may remain, but normalized internally to `teacher_lora_config`.

Required tests:
- CLI overrides for each new arg.
- env default fallback for each new arg.
- invalid LR rejected.

Task gate:
```bash
uv run pytest tests/test_distillation_epoch_config.py -q
```
Then Global Gates.

---

### QLR-030: Teacher Adapter Wiring + Trainability Surface
Status: [ ]
Allowlist:
- `src/models/distillation.py`
- `tests/test_phase4_semantics.py`

Required changes:
- In `QwenTeacherEncoder.__init__`, accept:
  - `teacher_lora_enabled: bool`
  - `teacher_lora_config_path: Path | None`
  - `teacher_lora_trainable: bool`
  - `teacher_lora_lr: float`
- Apply PEFT only when enabled.
- Freeze non-LoRA params.
- Set LoRA param `requires_grad` according to `teacher_lora_trainable`.
- Add helper:
  - `trainable_parameters() -> list[torch.nn.Parameter]`
- Add helper metadata accessors (or attributes) for checkpoint/provenance:
  - enabled/trainable/config path/lr/target modules.
- Log:
  - enabled/trainable/config path/lr
  - total params/trainable params.

Required tests:
- Adapter wrapping executed only when enabled (mocked loader/mocked peft wrapper).
- Trainable parameter count differs for trainable `true` vs `false`.
- No external model download in unit tests.

Task gate:
```bash
uv run pytest tests/test_phase4_semantics.py -q
```
Then Global Gates.

---

### QLR-040: Trainer Optimizer Wiring for Teacher LoRA
Status: [x]
Allowlist:
- `src/models/distillation.py`
- `tests/test_phase4_semantics.py`

Required changes:
- In `ContrastiveDistillationTrainer.__init__`:
  - Keep student optimizer path.
  - Add teacher optimizer param group only if `teacher.trainable_parameters()` non-empty.
  - Use `teacher_lora_lr` for teacher group.
- In `train_step`:
  - If teacher has trainable params, do teacher forward **without** `torch.no_grad()`.
  - If teacher has no trainable params, keep `no_grad()` path.
  - Ensure update step handles both student and teacher trainable params safely.

Required tests:
- Dummy teacher with trainable params: gradients present and step changes params.
- Dummy teacher without trainable params: no teacher grads, student still trains.

Task gate:
```bash
uv run pytest tests/test_phase4_semantics.py -q
```
Then Global Gates.

---

### QLR-050: Main Distillation Flow Integration
Status: [x]
Allowlist:
- `src/models/distillation.py`

Required changes:
- `main()` must pass new teacher-lora args into `QwenTeacherEncoder`.
- Preserve warmup embedding dimension inference and student build flow.
- Log a startup summary line with effective teacher LoRA settings.

Task gate:
```bash
uv run pytest tests/test_phase4_semantics.py tests/test_distillation_epoch_config.py -q
```
Then Global Gates.

---

### QLR-060: Retrain Propagation of Teacher LoRA Controls
Status: [ ]
Allowlist:
- `src/models/retrain.py`
- `tests/test_distillation_epoch_config.py`

Required changes:
- Extend retrain parse args with:
  - `--teacher-lora-enabled`
  - `--teacher-lora-config`
  - `--teacher-lora-trainable`
  - `--teacher-lora-lr`
- Extend `build_distillation_command(...)` to forward them.
- CPU fallback command must preserve these arguments and still append `--teacher-device cpu`.

Required tests:
- Command build includes all new args when supplied.
- Fallback command preserves args.

Task gate:
```bash
uv run pytest tests/test_distillation_epoch_config.py -q
```
Then Global Gates.

---

### QLR-070: Checkpoint Provenance Metadata
Status: [x]
Allowlist:
- `src/models/distillation.py`
- `tests/test_phase4_semantics.py`

Required changes:
- Extend distillation checkpoint payload with:
  - `teacher_lora_enabled`
  - `teacher_lora_trainable`
  - `teacher_lora_config_path`
  - `teacher_lora_lr`
  - `teacher_lora_target_modules`
- Keep existing checkpoint keys unchanged.

Required tests:
- Verify new metadata keys are present and correctly populated.

Task gate:
```bash
uv run pytest tests/test_phase4_semantics.py -q
```
Then Global Gates.

---

### QLR-080: Focused Runtime Micro-Checks
Status: [x]
Allowlist: none (no source edits)

Required commands:
```bash
uv run pytest tests/test_phase4_semantics.py tests/test_distillation_epoch_config.py -q
uv run python - <<'PY'
from src.models.distillation import parse_args
print(sorted(vars(parse_args([])).keys()))
PY
```

Expected keys include:
- `teacher_lora_enabled`
- `teacher_lora_config`
- `teacher_lora_trainable`
- `teacher_lora_lr`

Task gate:
- Run Global Gates.
- Stop.

---

### QLR-090: Regression Sweep
Status: [x]
Allowlist: none (no source edits)

Run:
```bash
uv run pytest tests/test_phase4_semantics.py tests/test_distillation_epoch_config.py tests/test_phase5_report_context.py tests/test_phase6_requirements.py -q
```

Task gate:
- Run Global Gates.
- Stop.

---

### QLR-100: Final Integrity Check
Status: [x]
Allowlist: none (no source edits)

Run:
```bash
rg -n "print\(" src tests
rg -n "teacher-lora|teacher_lora|get_peft_model|load_qlora_config|trainable_parameters" src/models/distillation.py src/models/retrain.py src/models/qlora_config.py
git diff -- src/models/qlora_config.py src/models/distillation.py src/models/retrain.py tests/test_phase4_semantics.py tests/test_distillation_epoch_config.py
```

Acceptance:
- No `print(` in src/tests.
- Required wiring symbols present.
- Diff limited to planned files.

Task gate:
- Run Global Gates.
- Stop and request final human sign-off.

---

## Blocker Handling
If blocked, append this section at file end and stop:

```markdown
## Blocker Report
- Task: QLR-XXX
- Command: <exact command>
- Error: <exact error>
- Minimal Root Cause: <one sentence>
- Proposed Recovery: <one sentence>
```

---

## Baseline Notes
- Task QLR-000: 20 tests passed in 5.69s (tests/test_phase4_semantics.py, tests/test_distillation_epoch_config.py).
