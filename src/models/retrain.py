from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from src.utils.logger import get_logger, setup_project

log = get_logger("retrain")

_FALLBACK_FREE_GIB_ENV = "DINOMALY_FALLBACK_FREE_GIB"
_FAILFAST_FREE_GIB_ENV = "DINOMALY_FAILFAST_FREE_GIB"
_DISTILLATION_EPOCHS_ENV = "DISTILLATION_EPOCHS"


@dataclass(frozen=True)
class GpuHeadroom:
    free_gib: float
    total_gib: float


@dataclass(frozen=True)
class DinomalyRuntimeProfile:
    mode: str
    train_batch_size: int
    eval_batch_size: int
    num_workers: int


def _read_numeric_setting(env_key: str, default_value: float) -> float:
    raw_value = os.getenv(env_key)
    if raw_value is None:
        return default_value
    try:
        value = float(raw_value)
    except ValueError as error:
        msg = f"Invalid numeric value for {env_key}: {raw_value!r}"
        raise RuntimeError(msg) from error
    if value <= 0:
        msg = f"{env_key} must be greater than 0, got {value}"
        raise RuntimeError(msg)
    return value


def _read_positive_int_setting(env_key: str, default_value: int) -> int:
    raw_value = os.getenv(env_key)
    if raw_value is None:
        return default_value
    try:
        value = int(raw_value)
    except ValueError as error:
        msg = f"Invalid integer value for {env_key}: {raw_value!r}"
        raise RuntimeError(msg) from error
    if value <= 0:
        msg = f"{env_key} must be greater than 0, got {value}"
        raise RuntimeError(msg)
    return value


def _positive_int(raw_value: str) -> int:
    value = int(raw_value)
    if value <= 0:
        msg = "Value must be a positive integer"
        raise argparse.ArgumentTypeError(msg)
    return value


def detect_gpu_headroom() -> GpuHeadroom | None:
    if not torch.cuda.is_available():
        log.info("CUDA is not available; skipping VRAM preflight policy")
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    gib = 1024**3
    return GpuHeadroom(
        free_gib=float(free_bytes) / gib,
        total_gib=float(total_bytes) / gib,
    )


def _extract_config_int(config_text: str, key: str) -> int:
    pattern = re.compile(
        rf"^(?P<prefix>\s*{re.escape(key)}:\s*)(?P<value>\d+)\s*$",
        re.MULTILINE,
    )
    match = pattern.search(config_text)
    if match is None:
        msg = f"Failed to resolve '{key}' in Dinomaly config"
        raise RuntimeError(msg)
    return int(match.group("value"))


def _replace_config_int(config_text: str, key: str, value: int) -> str:
    pattern = re.compile(
        rf"^(?P<prefix>\s*{re.escape(key)}:\s*)(?P<value>\d+)\s*$",
        re.MULTILINE,
    )

    def _replace(match: re.Match[str]) -> str:
        return f"{match.group('prefix')}{value}"

    updated_text, update_count = pattern.subn(_replace, config_text, count=1)
    if update_count != 1:
        msg = f"Failed to update '{key}' in Dinomaly config"
        raise RuntimeError(msg)
    return updated_text


def select_dinomaly_profile(
    config_text: str,
    headroom: GpuHeadroom | None,
) -> DinomalyRuntimeProfile:
    train_batch_size = _extract_config_int(
        config_text=config_text,
        key="train_batch_size",
    )
    eval_batch_size = _extract_config_int(
        config_text=config_text,
        key="eval_batch_size",
    )
    num_workers = _extract_config_int(config_text=config_text, key="num_workers")

    if headroom is None:
        return DinomalyRuntimeProfile(
            mode="default",
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
        )

    failfast_free_gib = _read_numeric_setting(
        env_key=_FAILFAST_FREE_GIB_ENV,
        default_value=1.0,
    )
    fallback_free_gib = _read_numeric_setting(
        env_key=_FALLBACK_FREE_GIB_ENV,
        default_value=4.0,
    )

    if fallback_free_gib < failfast_free_gib:
        msg = (
            f"{_FALLBACK_FREE_GIB_ENV} ({fallback_free_gib}) must be >= "
            f"{_FAILFAST_FREE_GIB_ENV} ({failfast_free_gib})"
        )
        raise RuntimeError(msg)

    if headroom.free_gib < failfast_free_gib:
        message = (
            "Insufficient free VRAM before Dinomaly training. "
            f"Detected free={headroom.free_gib:.2f} GiB, "
            f"total={headroom.total_gib:.2f} GiB, "
            f"required>={failfast_free_gib:.2f} GiB. "
            "Action: free GPU memory, reduce background GPU load, or set "
            f"{_FAILFAST_FREE_GIB_ENV}/{_FALLBACK_FREE_GIB_ENV} for controlled retries."
        )
        raise RuntimeError(message)

    if headroom.free_gib >= fallback_free_gib:
        return DinomalyRuntimeProfile(
            mode="default",
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
        )

    severe_threshold_gib = (failfast_free_gib + fallback_free_gib) / 2
    if headroom.free_gib < severe_threshold_gib:
        target_batch_size = 2
        target_workers = 2
    else:
        target_batch_size = 4
        target_workers = 4

    return DinomalyRuntimeProfile(
        mode="fallback",
        train_batch_size=min(train_batch_size, target_batch_size),
        eval_batch_size=min(eval_batch_size, target_batch_size),
        num_workers=min(num_workers, target_workers),
    )


def apply_dinomaly_profile(
    config_text: str,
    profile: DinomalyRuntimeProfile,
) -> str:
    updated_text = _replace_config_int(
        config_text=config_text,
        key="train_batch_size",
        value=profile.train_batch_size,
    )
    updated_text = _replace_config_int(
        config_text=updated_text,
        key="eval_batch_size",
        value=profile.eval_batch_size,
    )
    return _replace_config_int(
        config_text=updated_text,
        key="num_workers",
        value=profile.num_workers,
    )


def resolve_dinomaly_config(config_path: Path) -> Path:
    config_text = config_path.read_text(encoding="utf-8")
    headroom = detect_gpu_headroom()
    profile = select_dinomaly_profile(config_text=config_text, headroom=headroom)

    if profile.mode == "default":
        if headroom is not None:
            log.info(
                "Using default Dinomaly profile",
                free_gib=round(headroom.free_gib, 3),
                total_gib=round(headroom.total_gib, 3),
                train_batch_size=profile.train_batch_size,
                eval_batch_size=profile.eval_batch_size,
                num_workers=profile.num_workers,
            )
        return config_path

    runtime_config = apply_dinomaly_profile(config_text=config_text, profile=profile)
    runtime_config_dir = Path("results/runtime")
    runtime_config_dir.mkdir(parents=True, exist_ok=True)
    runtime_config_path = runtime_config_dir / "dinomaly_config.runtime.yaml"
    runtime_config_path.write_text(runtime_config, encoding="utf-8")
    if headroom is None:
        msg = "GPU headroom is required for fallback profile selection"
        raise RuntimeError(msg)
    log.warning(
        "Low VRAM detected; using fallback Dinomaly profile",
        free_gib=round(headroom.free_gib, 3),
        total_gib=round(headroom.total_gib, 3),
        runtime_config=str(runtime_config_path),
        train_batch_size=profile.train_batch_size,
        eval_batch_size=profile.eval_batch_size,
        num_workers=profile.num_workers,
    )
    return runtime_config_path


def build_anomalib_command(config_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.utils.anomalib_entrypoint",
        "train",
        "--config",
        str(config_path),
    ]


def build_distillation_command(
    distillation_epochs: int,
    teacher_device: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.models.distillation",
        "--epochs",
        str(distillation_epochs),
    ]
    if teacher_device is not None:
        command.extend(["--teacher-device", teacher_device])
    return command


def _signal_name_from_return_code(return_code: int) -> str | None:
    if return_code >= 0:
        return None

    signal_number = -return_code
    try:
        return signal.Signals(signal_number).name
    except ValueError:
        return f"SIG{signal_number}"


def _run_streaming_command(command: list[str], step_name: str) -> None:
    log.info("Starting retrain step", step=step_name, command=" ".join(command))
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        msg = "Failed to open subprocess stdout stream"
        raise RuntimeError(msg)
    with process.stdout:
        for line in process.stdout:
            sys.stdout.write(line)

    return_code = process.wait()
    elapsed_seconds = time.monotonic() - started
    if return_code != 0:
        signal_name = _signal_name_from_return_code(return_code)
        log.error(
            "Retrain step failed",
            step=step_name,
            return_code=return_code,
            signal=signal_name,
            duration_seconds=round(elapsed_seconds, 3),
        )
        raise subprocess.CalledProcessError(returncode=return_code, cmd=command)
    log.info(
        "Retrain step completed",
        step=step_name,
        duration_seconds=round(elapsed_seconds, 3),
    )


def run_distillation_step(distillation_epochs: int) -> None:
    initial_command = build_distillation_command(
        distillation_epochs=distillation_epochs,
    )
    try:
        _run_streaming_command(command=initial_command, step_name="distillation")
        return
    except subprocess.CalledProcessError as error:
        if error.returncode != -int(signal.SIGSEGV):
            raise

    fallback_command = build_distillation_command(
        distillation_epochs=distillation_epochs,
        teacher_device="cpu",
    )
    log.warning(
        "Distillation exited with SIGSEGV; retrying on CPU teacher device",
        first_command=" ".join(initial_command),
        fallback_command=" ".join(fallback_command),
    )
    _run_streaming_command(
        command=fallback_command,
        step_name="distillation_cpu_fallback",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full retraining pipeline orchestrator"
    )
    parser.add_argument(
        "--dinomaly-config",
        type=Path,
        default=Path("dinomaly_config.yaml"),
        help="Path to Dinomaly training configuration file",
    )
    parser.add_argument(
        "--distillation-epochs",
        type=_positive_int,
        default=_read_positive_int_setting(_DISTILLATION_EPOCHS_ENV, default_value=1),
        help=(
            "Number of distillation epochs to run. "
            "Defaults to DISTILLATION_EPOCHS env var if set, otherwise 1."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    setup_project()

    log.info(
        "Starting full retraining pipeline",
        dinomaly_config=str(args.dinomaly_config),
        distillation_epochs=args.distillation_epochs,
    )
    _run_streaming_command(
        command=[sys.executable, "-m", "src.models.classifier"],
        step_name="classifier",
    )

    runtime_config = resolve_dinomaly_config(config_path=args.dinomaly_config)
    _run_streaming_command(
        command=build_anomalib_command(config_path=runtime_config),
        step_name="dinomaly",
    )

    run_distillation_step(distillation_epochs=args.distillation_epochs)
    log.info("Retraining pipeline completed")


if __name__ == "__main__":
    main()
