#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

chmod +x ./retrain.sh ./evaluate.sh

./retrain.sh
./evaluate.sh

uv run python -m src.models.onnx_serialization
uv run python -m src.models.migraphx_compile