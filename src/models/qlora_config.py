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


def save_qlora_config(
    output_path: Path,
    settings: QLoRASettings | None = None,
) -> Path:
    payload = qlora_config_payload(settings=settings)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("Saved QLoRA configuration", path=str(output_path))
    return output_path
