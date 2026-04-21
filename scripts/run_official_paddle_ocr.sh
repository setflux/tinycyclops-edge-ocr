#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${TINYCYCLOPS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OCR_DIR="$PROJECT_ROOT/third_party/hailo-apps/hailo_apps/python/standalone_apps/paddle_ocr"
VENV="${TINYCYCLOPS_VENV:-${TINY_CYCLOPS_VENV:-$PROJECT_ROOT/.venv}}"

INPUT="${1:-$OCR_DIR/ocr_img1.png}"
OUTPUT_DIR="${2:-$PROJECT_ROOT/runs/official_ocr_$(date +%Y%m%d_%H%M%S)}"

DET_HEF="${TINY_CYCLOPS_OCR_DET_HEF:-$OCR_DIR/ocr_det.hef}"
OCR_HEF="${TINY_CYCLOPS_OCR_HEF:-$OCR_DIR/ocr.hef}"

mkdir -p "$OUTPUT_DIR"

cd "$OCR_DIR"
"$VENV/bin/python" paddle_ocr.py \
  -n "$DET_HEF" \
  -n "$OCR_HEF" \
  -i "$INPUT" \
  --no-display \
  -o "$OUTPUT_DIR" \
  --log-level info

printf 'output_dir=%s\n' "$OUTPUT_DIR"
