#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

MIGRAPHX_SITE_DIR="${MIGRAPHX_PYTHON_PATH:-/opt/rocm/lib}"
if [[ -d "$MIGRAPHX_SITE_DIR" ]]; then
  export PYTHONPATH="$MIGRAPHX_SITE_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

run_export_locally() {
  uv run python3 -m src.models.onnx_serialization
  uv run python3 -m src.models.migraphx_compile
}

if uv run python3 -c "import migraphx" >/dev/null 2>&1; then
  run_export_locally
  exit 0
fi

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "Host MIGraphX bindings unavailable; falling back to docker compose export runtime."
  docker compose run --rm --entrypoint bash pipeline -lc \
    "set -euo pipefail; \
    uv run python3 -m src.models.onnx_serialization; \
    uv run python3 -m src.models.migraphx_compile"
  exit 0
fi

run_export_locally