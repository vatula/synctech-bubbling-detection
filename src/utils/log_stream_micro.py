from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

from src.utils.logger import get_logger, setup_project

log = get_logger("log_stream_micro")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit deterministic short-lived logs for streaming experiments."
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="training",
        help="Logical stage label for experiment output.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of progress steps to emit.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.3,
        help="Delay between progress updates.",
    )
    return parser.parse_args(argv)


def _emit_newline_progress(stage: str, steps: int, delay_seconds: float) -> None:
    for index in range(1, steps + 1):
        sys.stdout.write(f"{stage} newline step {index}/{steps}\n")
        sys.stdout.flush()
        time.sleep(delay_seconds)


def _emit_carriage_progress(stage: str, steps: int, delay_seconds: float) -> None:
    for index in range(1, steps + 1):
        sys.stdout.write(f"\r{stage} carriage step {index}/{steps}")
        sys.stdout.flush()
        time.sleep(delay_seconds)
    sys.stdout.write("\n")
    sys.stdout.flush()


def main() -> int:
    setup_project()
    args = parse_args()

    log.info(
        "Starting streaming micro experiment",
        stage=args.stage,
        steps=args.steps,
        delay_seconds=args.delay_seconds,
    )
    _emit_newline_progress(
        stage=args.stage,
        steps=args.steps,
        delay_seconds=args.delay_seconds,
    )
    _emit_carriage_progress(
        stage=args.stage,
        steps=args.steps,
        delay_seconds=args.delay_seconds,
    )

    # Emit known noisy lines to verify filters preserve behavior.
    sys.stdout.write("(null): No such file or directory\n")
    sys.stdout.write(
        "💡 Tip: For seamless cloud logging and experiment tracking, "
        "try installing [litlogger]\n"
    )
    sys.stdout.write(f"{args.stage} experiment complete\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
