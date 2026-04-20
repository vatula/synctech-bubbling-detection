from __future__ import annotations

from pathlib import Path

import pytest

from src.models.retrain import (
    GpuHeadroom,
    apply_dinomaly_profile,
    build_anomalib_command,
    select_dinomaly_profile,
)

_CONFIG_TEMPLATE = """
data:
  train_batch_size: 16
  eval_batch_size: 16
  num_workers: 16
""".strip()


def test_select_dinomaly_profile_uses_default_on_sufficient_headroom() -> None:
    profile = select_dinomaly_profile(
        config_text=_CONFIG_TEMPLATE,
        headroom=GpuHeadroom(free_gib=8.0, total_gib=24.0),
    )

    assert profile.mode == "default"
    assert profile.train_batch_size == 16
    assert profile.eval_batch_size == 16
    assert profile.num_workers == 16


def test_select_dinomaly_profile_uses_fallback_on_low_headroom() -> None:
    profile = select_dinomaly_profile(
        config_text=_CONFIG_TEMPLATE,
        headroom=GpuHeadroom(free_gib=2.5, total_gib=24.0),
    )

    assert profile.mode == "fallback"
    assert profile.train_batch_size <= 4
    assert profile.eval_batch_size <= 4
    assert profile.num_workers <= 4


def test_select_dinomaly_profile_fails_fast_on_critical_headroom() -> None:
    with pytest.raises(RuntimeError, match="Insufficient free VRAM"):
        select_dinomaly_profile(
            config_text=_CONFIG_TEMPLATE,
            headroom=GpuHeadroom(free_gib=0.25, total_gib=24.0),
        )


def test_apply_dinomaly_profile_updates_batches_and_workers() -> None:
    profile_text = apply_dinomaly_profile(
        config_text=_CONFIG_TEMPLATE,
        profile=select_dinomaly_profile(
            config_text=_CONFIG_TEMPLATE,
            headroom=GpuHeadroom(free_gib=3.0, total_gib=24.0),
        ),
    )

    assert "train_batch_size: 4" in profile_text
    assert "eval_batch_size: 4" in profile_text
    assert "num_workers: 4" in profile_text


def test_build_anomalib_command_uses_project_bootstrap_entrypoint() -> None:
    command = build_anomalib_command(config_path=Path("dinomaly_config.yaml"))

    assert command[1:4] == ["-m", "src.utils.anomalib_entrypoint", "train"]
    assert command[-2:] == ["--config", "dinomaly_config.yaml"]
