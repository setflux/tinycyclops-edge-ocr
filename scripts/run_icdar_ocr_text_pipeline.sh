#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${TINYCYCLOPS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${TINYCYCLOPS_VENV:-${TINY_CYCLOPS_VENV:-$PROJECT_ROOT/.venv}}"

cd "$PROJECT_ROOT"
exec "$VENV/bin/python" -m tinycyclops_ocr.cli "$@"
