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

run_export_in_compose() {
  docker compose --profile dev --progress plain run \
    --rm \
    --no-deps \
    -T \
    --entrypoint bash \
    pipeline-dev \
    -lc "set -euo pipefail; \
    export UV_PROJECT_ENVIRONMENT=\"\${UV_PROJECT_ENVIRONMENT:-/cache/uv/project-env}\"; \
    mkdir -p \"\$UV_PROJECT_ENVIRONMENT\"; \
    echo '[export] ONNX serialization started'; \
    python3 -m src.models.onnx_serialization; \
    echo '[export] MIGraphX compilation started'; \
    python3 -m src.models.migraphx_compile"
}

if uv run python3 -c "import migraphx" >/dev/null 2>&1; then
  run_export_locally
  exit 0
fi

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "Host MIGraphX bindings unavailable; falling back to docker compose export runtime."
  run_export_in_compose
  exit 0
fi

run_export_locally
