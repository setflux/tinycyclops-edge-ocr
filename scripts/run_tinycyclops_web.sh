#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${TINYCYCLOPS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${TINYCYCLOPS_VENV:-${TINY_CYCLOPS_VENV:-$PROJECT_ROOT/.venv}}"
HOST="${TINYCYCLOPS_WEB_HOST:-0.0.0.0}"
PORT="${TINYCYCLOPS_WEB_PORT:-18041}"

cd "$PROJECT_ROOT"
exec "$VENV/bin/python" -m uvicorn tinycyclops_ocr.web_app:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers 1
