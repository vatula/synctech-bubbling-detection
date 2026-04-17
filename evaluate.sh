#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ "${1:-}" == "--docker" ]]; then
  docker compose run --rm training uv run python -m src.models.evaluation
else
  uv run python -m src.models.evaluation
fi