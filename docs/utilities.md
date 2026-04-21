# Utility Scripts

This document describes optional command-line utilities for running OCR
workloads and inspecting generated outputs outside the web UI.

## Prerequisites

Run these utilities from the repository root after completing
`docs/installation.md`.

Required before OCR utilities can perform real inference:

- `TINYCYCLOPS_ROOT` points at the cloned repository, or commands are run from
  the repository root.
- `$TINYCYCLOPS_VENV` points at the Python environment created with
  `--system-site-packages`.
- HailoRT is installed and `hailortcli scan` sees the Hailo device.
- `third_party/hailo-apps` exists at the tested Hailo Apps revision.
- `ocr_det.hef` and `ocr.hef` exist under the Hailo PaddleOCR app directory, or
  `TINY_CYCLOPS_OCR_DET_HEF` / `TINY_CYCLOPS_OCR_HEF` point at compatible local
  HEF files.
- The input dataset or image path exists locally.

## OCR Text Pipeline

TinyCyclops wraps Hailo's official PaddleOCR detector and recognizer without
patching the vendored `third_party/hailo-apps` tree.

### Command

```bash
cd "$TINYCYCLOPS_ROOT"
./scripts/run_icdar_ocr_text_pipeline.sh --json
```

Useful options:

```bash
./scripts/run_icdar_ocr_text_pipeline.sh --limit 10 --json
./scripts/run_icdar_ocr_text_pipeline.sh --input /path/to/image-or-dir --json
./scripts/run_icdar_ocr_text_pipeline.sh --output-dir /path/to/output --json
```

### Memory WorkBlock Pipeline

The memory WorkBlock mode keeps the official HAILO detector and recognizer
flow, but runs through a memory-resident `ImageWork` / `WorkBlock` wrapper:

```bash
./scripts/run_icdar_ocr_workblock_pipeline.sh --limit 10 --json
```

This mode adds stage metrics to `summary.json` under `metrics` and releases each
image's resident frame after that image's recognition result has been collected.
Use this mode to compare HAILO scheduling, CPU crop/resize behavior, and
resident frame release behavior against the baseline official wrapper.

WorkBlock size and HEF batch size are intentionally separate:

```bash
./scripts/run_icdar_ocr_workblock_pipeline.sh \
  --work-block-size 20 \
  --hef-batch-size 1 \
  --json
```

### Example Full Run

Dataset:

```text
$TINYCYCLOPS_ROOT/data/icdar2015/test_images
```

Artifacts:

```text
$TINYCYCLOPS_ROOT/runs/icdar2015_ocr_text_full_20260413/summary.json
$TINYCYCLOPS_ROOT/runs/icdar2015_ocr_text_full_20260413/results.jsonl
$TINYCYCLOPS_ROOT/runs/icdar2015_ocr_text_full_20260413/detections.csv
$TINYCYCLOPS_ROOT/runs/icdar2015_ocr_text_full_20260413/full_text.txt
```

Summary:

```text
image_count=500
detection_count=3165
nonempty_image_count=472
elapsed_seconds=43.28542078300006
fps=11.551233439698251
```

### Result Shape

`results.jsonl` has one record per image:

```json
{
  "image": "/absolute/path/to/img_1.jpg",
  "image_name": "img_1.jpg",
  "width": 1280,
  "height": 720,
  "text": "recognized text joined at image level",
  "detections": [
    {
      "index": 0,
      "box": {"x": 483, "y": 376, "w": 14, "h": 8},
      "text": "150",
      "confidence": 0.3686274509803922,
      "raw_confidence": 94.0
    }
  ]
}
```

`confidence` is normalized to 0.0-1.0 for web/API use. `raw_confidence` keeps
the official decoder's raw Hailo output scale.

### Notes

The official crop helper may print `Error with box` for some warped text boxes.
TinyCyclops captures that stdout into `run_stdout.log` for new runs so JSON
stdout remains machine-readable.
