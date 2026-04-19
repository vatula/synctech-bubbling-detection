import signal
import subprocess

import pytest

from src.models import retrain as retrain_module
from src.models.distillation import parse_args as parse_distillation_args
from src.models.retrain import parse_args as parse_retrain_args


def test_distillation_cli_epochs_override() -> None:
    args = parse_distillation_args(["--epochs", "5"])
    assert args.epochs == 5


def test_distillation_cli_epochs_uses_env_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISTILLATION_EPOCHS", "7")
    args = parse_distillation_args([])
    assert args.epochs == 7


def test_distillation_cli_epochs_rejects_non_positive() -> None:
    with pytest.raises(SystemExit):
        parse_distillation_args(["--epochs", "0"])


def test_distillation_cli_teacher_device_override() -> None:
    args = parse_distillation_args(["--teacher-device", "cpu"])
    assert args.teacher_device == "cpu"


def test_distillation_cli_teacher_device_uses_env_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISTILLATION_TEACHER_DEVICE", "cuda")
    args = parse_distillation_args([])
    assert args.teacher_device == "cuda"


def test_distillation_cli_teacher_device_invalid_env_falls_back_to_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISTILLATION_TEACHER_DEVICE", "gpu")
    args = parse_distillation_args([])
    assert args.teacher_device == "auto"


def test_retrain_cli_distillation_epochs_override() -> None:
    args = parse_retrain_args(["--distillation-epochs", "9"])
    assert args.distillation_epochs == 9


def test_retrain_cli_distillation_epochs_uses_env_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISTILLATION_EPOCHS", "3")
    args = parse_retrain_args([])
    assert args.distillation_epochs == 3


def test_retrain_cli_distillation_epochs_rejects_non_positive() -> None:
    with pytest.raises(SystemExit):
        parse_retrain_args(["--distillation-epochs", "0"])


def test_run_distillation_step_retries_on_sigsegv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], str]] = []

    def fake_run_streaming_command(command: list[str], step_name: str) -> None:
        calls.append((command, step_name))
        if len(calls) == 1:
            raise subprocess.CalledProcessError(
                returncode=-int(signal.SIGSEGV),
                cmd=command,
            )

    monkeypatch.setattr(
        retrain_module,
        "_run_streaming_command",
        fake_run_streaming_command,
    )

    retrain_module.run_distillation_step(distillation_epochs=5, student_architecture="fastvit_t8")

    assert len(calls) == 2
    first_command, first_step = calls[0]
    second_command, second_step = calls[1]
    assert first_step == "distillation"
    assert second_step == "distillation_cpu_fallback"
    assert "--teacher-device" not in first_command
    assert second_command[-2:] == ["--teacher-device", "cpu"]


def test_run_distillation_step_propagates_non_sigsegv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], str]] = []

    def fake_run_streaming_command(command: list[str], step_name: str) -> None:
        calls.append((command, step_name))
        raise subprocess.CalledProcessError(returncode=1, cmd=command)

    monkeypatch.setattr(
        retrain_module,
        "_run_streaming_command",
        fake_run_streaming_command,
    )

    with pytest.raises(subprocess.CalledProcessError):
        retrain_module.run_distillation_step(distillation_epochs=5, student_architecture="fastvit_t8")

    assert len(calls) == 1
