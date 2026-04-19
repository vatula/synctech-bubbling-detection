#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "$ROOT_DIR/pyproject.toml" || ! -d "$ROOT_DIR/src" ]]; then
  printf 'cleanup.sh: project root validation failed at %s\n' "$ROOT_DIR" >&2
  exit 1
fi

remove_target() {
  local target_path="$1"
  local absolute_path
  absolute_path="$(realpath -m "$ROOT_DIR/$target_path")"

  if [[ "$absolute_path" == "$ROOT_DIR" || "$absolute_path" == "/" ]]; then
    printf 'cleanup.sh: refusing to remove unsafe path %s\n' "$absolute_path" >&2
    exit 1
  fi

  if [[ "$absolute_path" == "$ROOT_DIR/.cache" || "$absolute_path" == "$ROOT_DIR/.cache/"* ]]; then
    printf 'cleanup.sh: refusing to remove protected cache path %s\n' "$absolute_path" >&2
    exit 1
  fi

  if [[ "$absolute_path" != "$ROOT_DIR/"* ]]; then
    printf 'cleanup.sh: refusing to remove path outside project root %s\n' "$absolute_path" >&2
    exit 1
  fi

  if [[ -e "$absolute_path" ]]; then
    rm -rf "$absolute_path"
    printf 'cleanup.sh: removed %s\n' "$target_path"
  fi
}

remove_target "results"
remove_target "pipeline_metrics_report.md"

shopt -s nullglob globstar
for pycache_dir in __pycache__ src/**/__pycache__ tests/**/__pycache__; do
  remove_target "$pycache_dir"
done
shopt -u nullglob globstar

remove_target ".pytest_cache"

mkdir -p "$ROOT_DIR/results"
printf 'cleanup.sh: cleanup complete (preserved .cache)\n'