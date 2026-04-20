import subprocess
from pathlib import Path

# Path to the validator script
VALIDATOR_SCRIPT = Path("scripts/validate_bugfix_checklist.py")


def run_validator(content: str) -> subprocess.CompletedProcess[str]:
    # Create a temporary checklist file
    temp_file = Path("temp_checklist.md")
    temp_file.write_text(content)
    try:
        return subprocess.run(
            ["python", str(VALIDATOR_SCRIPT), "--file", str(temp_file)],
            capture_output=True,
            text=True,
        )
    finally:
        temp_file.unlink()


def test_valid_checklist() -> None:
    content = """
### B3-000: ...
Status: [x]
Evidence:
- Done.
### B3-010: ...
Status: [ ]
"""
    assert run_validator(content).returncode == 0


def test_invalid_order() -> None:
    content = """
### B3-000: ...
Status: [ ]
### B3-010: ...
Status: [x]
"""
    assert run_validator(content).returncode != 0


def test_multiple_in_progress() -> None:
    content = """
### B3-000: ...
Status: [~]
### B3-010: ...
Status: [~]
"""
    assert run_validator(content).returncode != 0


def test_unknown_status() -> None:
    content = """
### B3-000: ...
Status: [?]
"""
    assert run_validator(content).returncode != 0


def test_missing_evidence_fails() -> None:
    content = """
### B4-000: ...
Status: [x]
Evidence:
- Pending.
"""
    assert run_validator(content).returncode != 0


def test_speculative_evidence_fails() -> None:
    content = """
### B4-000: ...
Status: [x]
Evidence:
- This is assumed to work.
"""
    assert run_validator(content).returncode != 0
