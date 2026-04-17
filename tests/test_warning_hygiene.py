from __future__ import annotations

import sys
import warnings
from typing import Any

import pytest

from src.utils.warning_hygiene import (
    WarningHygieneStream,
    get_warning_filter_rules,
    install_warning_hygiene,
    reset_warning_hygiene_for_tests,
)


def test_install_warning_hygiene_registers_filters() -> None:
    reset_warning_hygiene_for_tests()
    install_warning_hygiene()

    rule_keys = {rule.key for rule in get_warning_filter_rules()}
    assert "xformers_fallback" in rule_keys
    assert "lightning_pytree_deprecation" in rule_keys
    assert "predict_dataloader_workers_advisory" in rule_keys
    assert isinstance(sys.stdout, WarningHygieneStream)
    assert isinstance(sys.stderr, WarningHygieneStream)

    reset_warning_hygiene_for_tests()


def test_allowlisted_warnings_emit_one_time_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, Any]] = []

    class _Logger:
        def info(self, event: str, **kwargs: object) -> None:
            events.append({"event": event, **kwargs})

    reset_warning_hygiene_for_tests()
    monkeypatch.setattr("src.utils.warning_hygiene._LOG", _Logger())
    install_warning_hygiene()

    warnings.warn("xFormers is not available (Attention)", UserWarning, stacklevel=2)
    warnings.warn("xFormers is not available (Attention)", UserWarning, stacklevel=2)

    xformers_events = [
        event for event in events if event.get("key") == "xformers_fallback"
    ]
    assert len(xformers_events) == 1

    reset_warning_hygiene_for_tests()
