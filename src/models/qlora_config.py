from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from peft import LoraConfig, TaskType

from src.utils.logger import get_logger

log = get_logger("qlora_config")


@dataclass(frozen=True)
class QLoRASettings:
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: tuple[str, str, str, str] = field(
        default=("q_proj", "k_proj", "v_proj", "o_proj")
    )


def build_lora_config(settings: QLoRASettings | None = None) -> LoraConfig:
    effective = settings or QLoRASettings()
    task_type = TaskType[effective.task_type]
    config = LoraConfig(
        r=effective.r,
        lora_alpha=effective.lora_alpha,
        lora_dropout=effective.lora_dropout,
        bias=effective.bias,
        task_type=task_type,
        target_modules=list(effective.target_modules),
    )
    log.info(
        "Built QLoRA config",
        r=effective.r,
        lora_alpha=effective.lora_alpha,
        targets=list(effective.target_modules),
    )
    return config


def qlora_config_payload(settings: QLoRASettings | None = None) -> dict[str, Any]:
    effective = settings or QLoRASettings()
    payload: dict[str, Any] = {
        "peft_type": "LORA",
        "task_type": effective.task_type,
        "r": effective.r,
        "lora_alpha": effective.lora_alpha,
        "lora_dropout": effective.lora_dropout,
        "bias": effective.bias,
        "target_modules": list(effective.target_modules),
    }
    return payload


def qlora_settings_from_payload(payload: dict[str, Any]) -> QLoRASettings:
    # Strict validation
    required_keys = {
        "task_type",
        "r",
        "lora_alpha",
        "lora_dropout",
        "bias",
        "target_modules",
    }
    missing = required_keys - payload.keys()
    if missing:
        msg = f"Missing required QLoRA payload keys: {missing}"
        log.error(msg, missing=list(missing))
        raise ValueError(msg)

    r = payload["r"]
    if not isinstance(r, int) or r <= 0:
        msg = "r must be an integer > 0"
        log.error(msg, r=r)
        raise ValueError(msg)

    lora_alpha = payload["lora_alpha"]
    if not isinstance(lora_alpha, int) or lora_alpha <= 0:
        msg = "lora_alpha must be an integer > 0"
        log.error(msg, lora_alpha=lora_alpha)
        raise ValueError(msg)

    lora_dropout = payload["lora_dropout"]
    if not isinstance(lora_dropout, (float, int)) or not (0.0 <= lora_dropout < 1.0):
        msg = "lora_dropout must be a numeric value in [0.0, 1.0)"
        log.error(msg, lora_dropout=lora_dropout)
        raise ValueError(msg)

    bias = payload["bias"]
    if bias not in ["none", "all", "lora_only"]:
        msg = "bias must be one of 'none', 'all', 'lora_only'"
        log.error(msg, bias=bias)
        raise ValueError(msg)

    target_modules = payload["target_modules"]
    required_modules = {"q_proj", "k_proj", "v_proj", "o_proj"}
    if not isinstance(target_modules, (list, tuple)) or not required_modules.issubset(
        set(target_modules)
    ):
        msg = f"target_modules must include all: {required_modules}"
        log.error(msg, target_modules=target_modules)
        raise ValueError(msg)

    log.info("Successfully validated QLoRA payload")
    return QLoRASettings(
        r=int(r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        bias=bias,  # type: ignore
        task_type=str(payload["task_type"]),
        target_modules=tuple(target_modules),  # type: ignore
    )


def load_qlora_config(path: Path) -> QLoRASettings:
    if not path.exists():
        log.error("QLoRA config path not found", path=str(path))
        raise FileNotFoundError(f"QLoRA config not found at {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    return qlora_settings_from_payload(payload)


def save_qlora_config(
    output_path: Path,
    settings: QLoRASettings | None = None,
) -> Path:
    payload = qlora_config_payload(settings=settings)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("Saved QLoRA configuration", path=str(output_path))
    return output_path
