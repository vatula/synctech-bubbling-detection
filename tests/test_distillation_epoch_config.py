import signal
import subprocess
from pathlib import Path

import pytest

from src.models import retrain as retrain_module
from src.models.distillation import parse_args as parse_distillation_args
from src.models.retrain import (
    build_distillation_command,
    parse_args as parse_retrain_args,
)


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

    retrain_module.run_distillation_step(
        distillation_epochs=5, student_architecture="fastvit_t8"
    )

    assert len(calls) == 2
    _, first_step = calls[0]
    second_command, second_step = calls[1]
    assert first_step == "distillation"
    assert second_step == "distillation_cpu_fallback"
    assert "--teacher-device" in second_command
    assert second_command[second_command.index("--teacher-device") + 1] == "cpu"
    assert "--teacher-lora-lr" in second_command


def test_distillation_cli_teacher_lora_args() -> None:
    args = parse_distillation_args(
        [
            "--teacher-lora-enabled",
            "true",
            "--teacher-lora-config",
            "test.json",
            "--teacher-lora-trainable",
            "true",
            "--teacher-lora-lr",
            "1e-4",
        ]
    )
    assert args.teacher_lora_enabled is True
    assert args.teacher_lora_config == Path("test.json")
    assert args.teacher_lora_trainable is True
    assert args.teacher_lora_lr == 1e-4


def test_distillation_cli_teacher_lora_env_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISTILLATION_TEACHER_LORA_ENABLED", "true")
    monkeypatch.setenv("DISTILLATION_TEACHER_LORA_CONFIG", "env.json")
    monkeypatch.setenv("DISTILLATION_TEACHER_LORA_TRAINABLE", "true")
    monkeypatch.setenv("DISTILLATION_TEACHER_LORA_LR", "1e-3")
    args = parse_distillation_args([])
    assert args.teacher_lora_enabled is True
    assert args.teacher_lora_config == Path("env.json")
    assert args.teacher_lora_trainable is True
    assert args.teacher_lora_lr == 1e-3


def test_distillation_cli_teacher_lora_invalid_lr() -> None:
    with pytest.raises(SystemExit):
        parse_distillation_args(["--teacher-lora-lr", "invalid"])


def test_build_distillation_command_lora_args() -> None:
    cmd = build_distillation_command(
        distillation_epochs=5,
        student_architecture="fastvit_t8",
        teacher_lora_enabled=True,
        teacher_lora_config_path=Path("test.json"),
        teacher_lora_trainable=True,
        teacher_lora_lr=1e-4,
    )
    assert "--teacher-lora-enabled" in cmd
    assert "true" in cmd[cmd.index("--teacher-lora-enabled") + 1]
    assert "--teacher-lora-config" in cmd
    assert "test.json" in cmd[cmd.index("--teacher-lora-config") + 1]
    assert "--teacher-lora-trainable" in cmd
    assert "true" in cmd[cmd.index("--teacher-lora-trainable") + 1]
    assert "--teacher-lora-lr" in cmd
    assert "0.0001" in cmd[cmd.index("--teacher-lora-lr") + 1]
