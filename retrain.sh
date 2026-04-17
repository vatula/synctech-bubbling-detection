#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ "${1:-}" == "--docker" ]]; then
  docker compose run --rm training bash -lc "set -euo pipefail; uv run python -m src.models.classifier && uv run anomalib train --config dinomaly_config.yaml && uv run python -m src.models.distillation" 2>&1
else
  (
    uv run python -m src.models.classifier && \
      uv run anomalib train --config dinomaly_config.yaml && \
      uv run python -m src.models.distillation
  ) 2>&1
fi | awk '
  /^\(null\): No such file or directory$/ { next }
  /^💡 Tip: For seamless cloud logging and experiment tracking, try installing \[litlogger\]/ { next }
  { print }
'