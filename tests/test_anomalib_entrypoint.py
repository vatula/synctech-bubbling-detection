from __future__ import annotations

from types import ModuleType

import pytest

from src.utils import anomalib_entrypoint


def test_run_anomalib_main_bootstraps_project_before_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def _fake_setup_project() -> None:
        events.append("setup_project")

    def _fake_anomalib_main() -> None:
        events.append("anomalib_main")

    monkeypatch.setattr(anomalib_entrypoint, "setup_project", _fake_setup_project)
    anomalib_entrypoint.run_anomalib_main(anomalib_main=_fake_anomalib_main)

    assert events == ["setup_project", "anomalib_main"]


def test_main_uses_lazy_anomalib_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class _CliModule:
        @staticmethod
        def main() -> None:
            events.append("anomalib_main")

    def _fake_import_module(name: str) -> ModuleType:
        assert name == "anomalib.cli.cli"
        return _CliModule  # type: ignore[return-value]

    monkeypatch.setattr(anomalib_entrypoint, "_import_module", _fake_import_module)

    def _fake_setup_project() -> None:
        events.append("setup_project")

    monkeypatch.setattr(anomalib_entrypoint, "setup_project", _fake_setup_project)

    anomalib_entrypoint.main()

    assert events == ["setup_project", "anomalib_main"]
