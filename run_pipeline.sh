#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

chmod +x ./retrain.sh ./evaluate.sh ./export.sh

./retrain.sh
./evaluate.sh
./export.sh