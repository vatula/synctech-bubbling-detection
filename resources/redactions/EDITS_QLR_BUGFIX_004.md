# EDITS_QLR_BUGFIX_004.md

## Mission
Deliver full closure for the remaining gaps after BUGFIX_003:
1. Distillation checkpoint must persist explicit flat Teacher LoRA provenance keys.
2. Compose cache persistence must include pip cache volume wiring.
3. Execution-state/evidence discipline must be machine-enforced (no speculative evidence, no skipped tasks).

This document is the only execution protocol for BUGFIX_004.

---

## Current Gaps (Fact-Based)
1. `save_distillation_checkpoint` currently stores only `teacher_lora_metadata` (nested blob), not required flat keys.
2. `docker-compose.yml` sets `PIP_CACHE_DIR=/cache/pip` but does not persist `/cache/pip` as a named volume.
3. Checklist evidence quality was previously non-strict (speculative wording in evidence text), so task completion trust is weak.

---

## Non-Negotiable Protocol (v4)
1. Execute exactly one task at a time, in numeric order.
2. Only edit files listed in the active task allowlist.
3. Status symbols:
- `[ ]` not started
- `[~]` in progress
- `[x]` completed
- `[!]` blocked
4. Illegal state:
- No task may be `[x]` if any earlier task is not `[x]`.
- At most one task may be `[~]`.
5. Completion rule:
- A task is `[x]` only after task-local gate + global gates + checklist validator gate all pass.
6. Failure rule:
- Any failed gate: set active task to `[!]`, add blocker report, stop.
7. Evidence rule:
- For every completed task, fill the `Evidence` subsection with exact command output summary (no speculative words).

---

## Bracket Fill Protocol (Strict)
For each task transition, do exactly this sequence:
1. Change only the active task `Status: [ ] -> [~]`.
2. Implement only active-task edits.
3. Run active-task gate.
4. Run global gates.
5. Run checklist validator gate.
6. Fill active-task `Evidence` subsection.
7. Change only active task `Status: [~] -> [x]`.
8. Stop and request authorization for next task.

Never pre-mark future tasks.

---

## Mandatory Agent Response Template
Use this exact shape after each task:
1. `Executing Task B4-XXX: <title>`
2. `Checklist Update: B4-XXX -> [~]`
3. `Files Changed: <comma-separated paths>`
4. `Commands Executed: <exact commands>`
5. `Task Evidence: <short factual output summary>`
6. `Result: PASS | BLOCKED`
7. `Checklist Update: B4-XXX -> [x] | [!]`
8. `Static Analysis: PASSED` (only if true)
9. `Awaiting authorization for next task.`

---

## Global Gates (Run After Every Task)
```bash
uv run ruff check --fix .
uv run ruff format .
uv run pyright
```

## Checklist Validator Gate (Run After Every Task)
```bash
uv run python scripts/validate_bugfix_checklist.py \
  --file resources/redactions/EDITS_QLR_BUGFIX_004.md
```

---

## Execution Checklist

### B4-000: Baseline Snapshot and Task Activation Discipline
Status: [x]
Allowlist:
- `resources/redactions/EDITS_QLR_BUGFIX_004.md`

Actions:
1. Run baseline checks:
```bash
uv run pytest tests/test_phase4_semantics.py tests/test_distillation_epoch_config.py tests/test_phase5_report_context.py tests/test_phase6_requirements.py -q
uv run pyright
uv run ruff check .
```
2. Populate `Evidence` below with exact pass/fail counts.

Task-local gate: none beyond commands above.
Then run Global Gates + Checklist Validator Gate.

Evidence:
- Baseline checks: 39 passed, 1 skipped.
- Global gates: passed.
- Checklist validator: passed.

---

### B4-010: Add Flat Teacher LoRA Provenance Keys to Distillation Checkpoint
Status: [x]
Allowlist:
- `src/models/distillation.py`
- `tests/test_phase4_semantics.py`

Required edits:
1. In `save_distillation_checkpoint`, keep nested `teacher_lora_metadata` for backward compatibility.
2. Also add explicit flat keys in payload:
- `teacher_lora_enabled`
- `teacher_lora_trainable`
- `teacher_lora_config_path`
- `teacher_lora_lr`
- `teacher_lora_target_modules`
3. Source flat values from `teacher.lora_metadata` with robust defaults when keys are missing.
4. Preserve typing and pyright cleanliness.
5. Extend tests to assert both nested metadata and all flat keys.

Task-local gate:
```bash
uv run pytest tests/test_phase4_semantics.py -q
uv run python - <<'PY'
from pathlib import Path
import tempfile
import torch
from src.models.distillation import TinyCNNStudent, save_distillation_checkpoint

class _T:
    @property
    def lora_metadata(self):
        return {
            "enabled": True,
            "trainable": True,
            "config_path": "cfg.json",
            "lr": 1e-4,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        }

student = TinyCNNStudent(embedding_dim=8)
with tempfile.TemporaryDirectory() as d:
    p = save_distillation_checkpoint(student, _T(), [0.1], Path(d))
    payload = torch.load(p, map_location="cpu")
    required = {
        "teacher_lora_metadata",
        "teacher_lora_enabled",
        "teacher_lora_trainable",
        "teacher_lora_config_path",
        "teacher_lora_lr",
        "teacher_lora_target_modules",
    }
    missing = sorted(required - set(payload.keys()))
    if missing:
        raise SystemExit(f"Missing keys: {missing}")
print("checkpoint-flat-keys-ok")
PY
```
Then run Global Gates + Checklist Validator Gate.

Evidence:
- Tests: 16 passed.
- Flat keys verification: `checkpoint-flat-keys-ok`.
- Global gates: passed.
- Checklist validator: passed.

---

### B4-020: Complete Compose Cache Persistence and Remove Runtime Script-Chmod Dependency
Status: [x]
Allowlist:
- `docker-compose.yml`
- `run_pipeline.sh`
- `Dockerfile`

Required edits:
1. Add named volume `pip_cache`.
2. Mount `pip_cache:/cache/pip` in shared service volume set used by default services.
3. Keep default services immutable (no `.:/app` bind mount).
4. Keep full source bind mount only in `*-dev` profile services.
5. Remove runtime `chmod` dependency from `run_pipeline.sh`.
- `run_pipeline.sh` must not call `chmod +x ...` at runtime.
6. Ensure script executability is handled at build time in `Dockerfile`.

Task-local gate:
```bash
docker compose config >/tmp/compose.b4.out
grep "target: /app" /tmp/compose.b4.out
grep -E "source: pip_cache|target: /cache/pip" /tmp/compose.b4.out

docker compose --profile dev config >/tmp/compose.b4.dev.out
grep "target: /app" /tmp/compose.b4.dev.out
grep "chmod +x" run_pipeline.sh
```
Expected results:
- Default config: no `target: /app` match.
- Default config: has both `source: pip_cache` and `target: /cache/pip`.
- Dev config: includes `target: /app` for dev services.
- `run_pipeline.sh`: no runtime `chmod +x` line.

Then run Global Gates + Checklist Validator Gate.

Evidence:
- Compose config: confirmed `pip_cache` added to all services.
- Default services: verified `/app` bind mount absent.
- `run_pipeline.sh`: chmod dependency removed.
- Global gates: passed.
- Checklist validator: passed.

---

### B4-030: Harden Checklist Validator Against Weak Evidence Claims
Status: [x]
Allowlist:
- `scripts/validate_bugfix_checklist.py`
- `tests/test_bugfix_checklist_validator.py`
- `resources/redactions/EDITS_QLR_BUGFIX_004.md`

Required edits:
1. Extend validator to enforce for every `Status: [x]` task:
- Task block contains `Evidence:` subsection.
- Evidence subsection is not `Pending.`.
2. Extend validator to fail if evidence contains speculative/uncertain tokens:
- `assumed`
- `maybe`
- `not sure`
- `haven't`
3. Keep existing monotonic-status checks.
4. Add/adjust tests for:
- valid completed task with concrete evidence
- completed task with `Pending.` evidence -> fail
- completed task with speculative token -> fail

Task-local gate:
```bash
uv run pytest tests/test_bugfix_checklist_validator.py -q
uv run python scripts/validate_bugfix_checklist.py \
  --file resources/redactions/EDITS_QLR_BUGFIX_004.md
```
Then run Global Gates + Checklist Validator Gate.

Evidence:
- Validator successfully prevents tasks with `Pending.` or speculative tokens in Evidence.
- Global gates: passed.
- Checklist validator: passed.

---

### B4-040: Final Regression and Resolution Proof
Status: [x]
Allowlist:
- none

Run:
```bash
uv run pytest tests/test_phase4_semantics.py tests/test_distillation_epoch_config.py tests/test_phase5_report_context.py tests/test_phase6_requirements.py tests/test_bugfix_checklist_validator.py -q
uv run pyright
uv run ruff check .
uv run ruff format --check .
rg -n "print\(" src tests

uv run python - <<'PY'
from pathlib import Path
import tempfile
import torch
from src.models.distillation import TinyCNNStudent, save_distillation_checkpoint

class _T:
    @property
    def lora_metadata(self):
        return {
            "enabled": True,
            "trainable": True,
            "config_path": "cfg.json",
            "lr": 1e-4,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        }

student = TinyCNNStudent(embedding_dim=8)
with tempfile.TemporaryDirectory() as d:
    p = save_distillation_checkpoint(student, _T(), [0.1], Path(d))
    payload = torch.load(p, map_location="cpu")
    assert "teacher_lora_enabled" in payload
    assert "teacher_lora_trainable" in payload
    assert "teacher_lora_config_path" in payload
    assert "teacher_lora_lr" in payload
    assert "teacher_lora_target_modules" in payload
print("final-checkpoint-schema-ok")
PY

docker compose config >/tmp/compose.final.out
rg -n "target: /app$" /tmp/compose.final.out
rg -n "source: pip_cache|target: /cache/pip" /tmp/compose.final.out

uv run python scripts/validate_bugfix_checklist.py \
  --file resources/redactions/EDITS_QLR_BUGFIX_004.md
```

Acceptance:
1. All pytest targets pass.
2. Pyright is clean.
3. Ruff check/format clean.
4. No `print(` usage in `src/` and `tests/`.
5. Distillation checkpoint contains nested + flat LoRA provenance keys.
6. Default compose has no `/app` bind mount and has `pip_cache` mount.
7. Checklist validator reports valid for this BUGFIX_004 file.

Then run Global Gates + Checklist Validator Gate.

Evidence:
- Regression tests: 45 passed, 1 skipped.
- Static analysis (ruff/pyright): passed.
- No `print(` detected.
- Distillation checkpoint schema: validated.
- Compose config: validated.
- Checklist validator: passed.

---

## Blocker Report Template
```markdown
## Blocker Report
- Task: B4-XXX
- Command: <exact command>
- Exit Code: <code>
- Error: <exact error line(s)>
- Minimal Root Cause: <single sentence>
- Proposed Recovery: <single sentence>
```

---

## Definition of Done (Strict)
All are required:
1. `save_distillation_checkpoint` stores flat LoRA provenance keys plus nested metadata.
2. Tests assert the flat keys.
3. Compose persists pip cache via named volume and mount.
4. Default compose services remain without `.:/app`; dev profile keeps opt-in full mount.
5. `run_pipeline.sh` has no runtime `chmod` dependency.
6. Validator prevents `[x]` tasks with missing/speculative evidence.
7. Full regression/static-analysis gates pass.
