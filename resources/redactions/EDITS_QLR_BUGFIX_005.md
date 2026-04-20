# EDITS_QLR_BUGFIX_005.md

## Mission
Close the final BUGFIX_004 gap by delivering a validator implementation and verification flow that are both internally consistent and externally reproducible.

Target outcomes:
1. Checklist validator rejects completed tasks with placeholder evidence like `- Pending.`.
2. Checklist validator continues rejecting speculative evidence tokens.
3. Full regression and static-analysis gates are clean.
4. Compose cache and checkpoint provenance requirements remain satisfied.

---

## Gap Analysis (RCA)
### RCA-1: Why BUGFIX_004 still failed
1. Evidence parsing in `scripts/validate_bugfix_checklist.py` used an exact equality check for `"Pending."`.
2. Real checklist evidence is typically bullet formatted (for example `- Pending.`), so the check did not trigger.
3. Result: `tests/test_bugfix_checklist_validator.py::test_missing_evidence_fails` failed while the checklist claimed completion.

### RCA-2: Why this caused a false completion state
1. BUGFIX_004 task statuses were already marked `[x]`.
2. Core validator behavior did not match required policy semantics.
3. Final gates were therefore inconsistent with checklist state.

---

## Non-Negotiable Protocol (v5)
1. Complete tasks in strict order.
2. A task reaches `[x]` only after task-local gates, global gates, and checklist validator gate pass.
3. Evidence text must be factual and command-derived.
4. If any gate fails, set `[!]`, log blocker, and stop progression.

Status symbols:
- `[ ]` not started
- `[~]` in progress
- `[x]` completed
- `[!]` blocked

---

## Global Gates
```bash
uv run ruff check --fix .
uv run ruff format .
uv run pyright
```

## Checklist Validator Gate
```bash
uv run python scripts/validate_bugfix_checklist.py \
  --file resources/redactions/EDITS_QLR_BUGFIX_005.md
```

---

## Execution Checklist

### B5-000: Reproduce and Pinpoint the Defect
Status: [x]
Allowlist:
- `scripts/validate_bugfix_checklist.py`
- `tests/test_bugfix_checklist_validator.py`

Commands:
```bash
uv run pytest tests/test_bugfix_checklist_validator.py -q
nl -ba scripts/validate_bugfix_checklist.py | sed -n '1,260p'
```

Evidence:
- Reproduction showed failing test `test_missing_evidence_fails`.
- Validator used `evidence_section == "Pending."`, which does not match bullet-form `- Pending.`.

---

### B5-010: Implement Validator Fix for Pending Bullet Evidence
Status: [x]
Allowlist:
- `scripts/validate_bugfix_checklist.py`

Required edits:
1. Normalize evidence lines by stripping bullet prefixes and whitespace.
2. Reject placeholder evidence when normalized line equals `pending` or `pending.`.
3. Preserve speculative token checks.
4. Keep code lint-clean.

Commands:
```bash
uv run pytest tests/test_bugfix_checklist_validator.py -q
uv run ruff check --fix scripts/validate_bugfix_checklist.py tests/test_bugfix_checklist_validator.py
uv run ruff format scripts/validate_bugfix_checklist.py tests/test_bugfix_checklist_validator.py
```

Evidence:
- Validator tests passed: `6 passed`.
- Ruff check passed on validator/test files.
- Ruff format reported no pending changes on those files.

---

### B5-020: Full Regression and Policy Compliance Sweep
Status: [x]
Allowlist:
- none

Commands:
```bash
uv run pytest tests/test_phase4_semantics.py tests/test_distillation_epoch_config.py tests/test_phase5_report_context.py tests/test_phase6_requirements.py tests/test_bugfix_checklist_validator.py -q
uv run pyright
uv run ruff check .
uv run ruff format --check .
rg -n "print\(" src tests

uv run python scripts/validate_bugfix_checklist.py \
  --file resources/redactions/EDITS_QLR_BUGFIX_004.md

docker compose config >/tmp/compose.b5.out
(rg -n "target: /app$" /tmp/compose.b5.out || true)
rg -n "source: pip_cache|target: /cache/pip" /tmp/compose.b5.out

docker compose --profile dev config >/tmp/compose.b5.dev.out
rg -n "target: /app$" /tmp/compose.b5.dev.out

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
    keys = [
        "teacher_lora_metadata",
        "teacher_lora_enabled",
        "teacher_lora_trainable",
        "teacher_lora_config_path",
        "teacher_lora_lr",
        "teacher_lora_target_modules",
    ]
    missing = [k for k in keys if k not in payload]
    print("missing", missing)
PY
```

Evidence:
- Regression suite result: `45 passed, 1 skipped`.
- Pyright result: `0 errors, 0 warnings, 0 informations`.
- Ruff check result: `All checks passed!`.
- Ruff format check result: `48 files already formatted`.
- `rg -n "print\(" src tests` returned no matches.
- BUGFIX_004 checklist validator result: `Checklist valid.`.
- Default compose output showed `pip_cache` mount lines and no `/app` bind mount.
- Dev compose output showed `/app` bind mount entries as expected.
- Checkpoint micro-test reported `missing []` for required LoRA provenance keys.

---

### B5-030: Validate This BUGFIX_005 Checklist
Status: [x]
Allowlist:
- `resources/redactions/EDITS_QLR_BUGFIX_005.md`

Commands:
```bash
uv run python scripts/validate_bugfix_checklist.py \
  --file resources/redactions/EDITS_QLR_BUGFIX_005.md
```

Evidence:
- Validator result: `Checklist valid.`.

---

## Final Resolution Statement
Resolution status: complete.

Closed gaps:
1. Validator now enforces placeholder evidence rejection for bullet-form pending content.
2. Validator test suite passes, including missing-evidence and speculative-evidence coverage.
3. Full regression/static analysis gates are clean.
4. Compose cache policy and checkpoint provenance requirements remain satisfied.
