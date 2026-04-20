from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import ModuleType

from src.utils.logger import setup_project


def _import_module(name: str) -> ModuleType:
    return import_module(name)


def run_anomalib_main(anomalib_main: Callable[[], None]) -> None:
    """Runs Anomalib CLI after project bootstrap is applied."""
    setup_project()
    anomalib_main()


def main() -> None:
    cli_module = _import_module("anomalib.cli.cli")
    anomalib_main = cli_module.main

    run_anomalib_main(anomalib_main=anomalib_main)


if __name__ == "__main__":
    main()
