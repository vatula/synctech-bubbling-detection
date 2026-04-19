#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ "${LOG_STREAM_MICRO_EXPERIMENT:-0}" == "1" ]]; then
  uv run python -m src.utils.log_stream_micro \
    --stage evaluate \
    --steps 5 \
    --delay-seconds "${LOG_STREAM_MICRO_DELAY_SECONDS:-0.2}" 2>&1
elif [[ "${1:-}" == "--docker" ]]; then
  docker compose run --rm training uv run python -m src.models.evaluation 2>&1
else
  uv run python -m src.models.evaluation 2>&1
fi | stdbuf -o0 tr '\r' '\n' | awk '
  /^\(null\): No such file or directory$/ { next }
  /^💡 Tip: For seamless cloud logging and experiment tracking, try installing \[litlogger\]/ { next }
  { print; fflush() }
'