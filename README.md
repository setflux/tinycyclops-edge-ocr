# TinyCyclops

TinyCyclops is a Raspberry Pi 5 + HAILO-8 OCR workload demo. It wraps HAILO's
official PaddleOCR HEFs with a TinyCyclops memory-resident WorkBlock pipeline
and a FastAPI web console.

This README assumes a Raspberry Pi 5 running a Debian/Raspberry Pi OS style
environment. It intentionally does not describe unrelated macOS, Windows, or
Rocky Linux installation paths.

## Repository Scope

The git repository contains TinyCyclops source code, helper scripts, docs, and
the web UI assets. Large runtime dependencies, HAILO official assets, datasets,
run outputs, and service files are intentionally outside git.

Tracked in git:

- `tinycyclops_ocr/`: TinyCyclops Python package.
- `scripts/`: local helper scripts.
- `web/`: FastAPI-served web console and static assets.
- `web/static/tinycyclops-logo.png`: repo-local logo asset.
- `docs/`: installation, dataset, and pipeline documentation.

Not tracked in git:

- `data/`: ICDAR/CCPD datasets and generated indexes.
- `runs/`: OCR artifacts, web run outputs, and temporary custom-upload workdirs.
- `third_party/`: HAILO official app checkout and downloaded HEF assets.
- `$TINYCYCLOPS_VENV` or `$TINYCYCLOPS_ROOT/.venv`: user-created Python runtime.
- `/etc/systemd/system/tinycyclops.service`: optional local service unit.
- Any reverse-proxy configuration.

## Quickstart

For a complete public-user installation guide, see
[`docs/installation.md`](docs/installation.md).

Additional docs:

- [`docs/demo_data_preparation.md`](docs/demo_data_preparation.md): dataset sources, layout, and validation notes.
- [`docs/utilities.md`](docs/utilities.md): optional CLI utility scripts, including the OCR text pipeline.

The short path is:

```bash
git clone https://github.com/YOUR_ORG/tinycyclops-edge-ocr.git
cd tinycyclops-edge-ocr
export TINYCYCLOPS_ROOT="$PWD"

# HailoRT and the Hailo device driver must already be installed.
hailortcli --version
hailortcli scan

python3 -m venv --system-site-packages .venv
export TINYCYCLOPS_VENV="$TINYCYCLOPS_ROOT/.venv"
. "$TINYCYCLOPS_VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements-web.txt

mkdir -p third_party
git clone https://github.com/hailo-ai/hailo-apps.git third_party/hailo-apps
cd third_party/hailo-apps
git checkout 891ce701c2ebe239a5d277759eb75a30f76678a9
cd hailo_apps/python/standalone_apps/paddle_ocr
python -m pip install -r requirements.txt
bash ./download_resources.sh --arch 8

cd "$TINYCYCLOPS_ROOT"
./scripts/run_official_paddle_ocr.sh
./scripts/run_tinycyclops_web.sh
```

Then check:

```bash
curl -s http://127.0.0.1:18041/health
```

## External Runtime Stack

TinyCyclops is not fully portable by cloning this repository alone. A target
Pi needs the following repo-external runtime pieces.

| Component | Default / User-configured Location | Purpose |
| --- | --- | --- |
| HAILO runtime | HailoRT `4.23.0`; `hailortcli --version` reports `HailoRT-CLI version 4.23.0` | HAILO device driver/runtime and Python bindings. |
| HAILO device | `hailortcli scan` should see the HAILO-8 device | Required for real HEF inference. |
| OCR venv | `$TINYCYCLOPS_VENV` or `$TINYCYCLOPS_ROOT/.venv` | Python runtime used by scripts, web server, and OCR child processes. |
| Python | `Python 3.13.5` in the OCR venv | Current tested Python version. |
| HAILO Apps checkout | `third_party/hailo-apps` | Official HAILO app code imported by TinyCyclops. |
| HAILO Apps revision | `891ce701c2ebe239a5d277759eb75a30f76678a9`, describe `26.03.1` | Tested upstream revision. |
| Official OCR HEFs | `third_party/hailo-apps/hailo_apps/python/standalone_apps/paddle_ocr/ocr_det.hef` and `ocr.hef` | Detection and recognition HEFs. |
| ICDAR data | `data/icdar2015` | Preset Workload 1. |
| CCPD data | `data/ccpd` | Preset Workload 2 randomized license-plate workload. |
| systemd unit | `/etc/systemd/system/tinycyclops.service` | Optional boot-time FastAPI web service. |

## Python Modules

The OCR venv was created with `--system-site-packages` so it can see HAILO
system packages. The active TinyCyclops/HAILO OCR modules are:

- `fastapi`, `starlette`, `uvicorn`, `pydantic`, `python-multipart`: web service and upload API.
- `opencv` / `cv2`, `numpy`: image loading, preprocessing, thumbnail generation, and pipeline data handling.
- `paddlepaddle`, `shapely`, `pyclipper`, `symspellpy`: official PaddleOCR/HAILO postprocess path.
- `python-dotenv`, `PyYAML`: official HAILO Apps support modules.
- `hailort`, `hailo_platform`, `hailo-tappas-core-python-binding`: HAILO runtime and official app bindings exposed through the system-visible Python stack.

Runtime checks:

```bash
export TINYCYCLOPS_ROOT="${TINYCYCLOPS_ROOT:-$PWD}"
export TINYCYCLOPS_VENV="${TINYCYCLOPS_VENV:-$TINYCYCLOPS_ROOT/.venv}"
"$TINYCYCLOPS_VENV/bin/python" --version
"$TINYCYCLOPS_VENV/bin/pip" list --format=freeze
hailortcli --version
hailortcli scan
```

The upstream standalone Paddle OCR requirement file lists:

```text
opencv-python<=4.10.0.84
numpy<2.0
shapely
pyclipper
symspellpy
python-dotenv
PyYAML
paddlepaddle
```

The reference environment currently runs with system-visible `numpy 2.2.4` and
`opencv 4.10.0`.
Smoke tests passed with this combination, so do not casually replace the system
HAILO/OpenCV/Numpy stack unless there is a concrete regression to fix.

## HAILO Apps And HEF Assets

`third_party/` is git-ignored. Recreate the official app tree on a new Pi:

```bash
export TINYCYCLOPS_ROOT="${TINYCYCLOPS_ROOT:-$PWD}"
cd "$TINYCYCLOPS_ROOT"
git clone https://github.com/hailo-ai/hailo-apps.git third_party/hailo-apps
cd third_party/hailo-apps
git checkout 891ce701c2ebe239a5d277759eb75a30f76678a9
```

Download the official HAILO-8 Paddle OCR resources with the upstream resource
script. The resulting assets must exist here:

```text
$TINYCYCLOPS_ROOT/third_party/hailo-apps/hailo_apps/python/standalone_apps/paddle_ocr/ocr_det.hef
$TINYCYCLOPS_ROOT/third_party/hailo-apps/hailo_apps/python/standalone_apps/paddle_ocr/ocr.hef
$TINYCYCLOPS_ROOT/third_party/hailo-apps/hailo_apps/python/standalone_apps/paddle_ocr/ocr_img1.png
$TINYCYCLOPS_ROOT/third_party/hailo-apps/hailo_apps/python/standalone_apps/paddle_ocr/ocr_img2.png
```

TinyCyclops defaults are defined in `tinycyclops_ocr/runtime.py`. HEF paths can
also be overridden for the official smoke wrapper:

```bash
export TINY_CYCLOPS_OCR_DET_HEF=/path/to/ocr_det.hef
export TINY_CYCLOPS_OCR_HEF=/path/to/ocr.hef
```

## Datasets

`data/` is git-ignored and must be prepared on each runtime host.

### ICDAR 2015

Preset Workload 1 expects the ICDAR 2015 Robust Reading Competition Challenge 4
test images:

```text
$TINYCYCLOPS_ROOT/data/icdar2015/ch4_test_images.zip
$TINYCYCLOPS_ROOT/data/icdar2015/Challenge4_Test_Task1_GT.zip
$TINYCYCLOPS_ROOT/data/icdar2015/test_images
$TINYCYCLOPS_ROOT/data/icdar2015/gt
```

Expected counts:

```text
test_images: 500 jpg files
gt: 500 gt_img_*.txt files
```

Source:

- RRC downloads page: `https://rrc.cvc.uab.es/?ch=4&com=downloads`
- Test images: `https://rrc.cvc.uab.es/downloads/ch4_test_images.zip`
- Ground truth: `https://rrc.cvc.uab.es/downloads/Challenge4_Test_Task1_GT.zip`

### CCPD2019

Preset Workload 2 uses CCPD2019 as a public license-plate source. TinyCyclops OCR
is currently alphanumeric-focused, so Chinese characters may be missed or
misread.

Source:

- Zenodo record: `https://zenodo.org/records/15647076`
- DOI: `10.5281/zenodo.15647076`
- License: Creative Commons Attribution 4.0 International
- Archive: `CCPD2019.tar.xz`
- Expected archive size: `13164924944` bytes
- Expected MD5: `0dfbca0e6fcb7cb8ea720b0eae94c735`

Expected local layout:

```text
$TINYCYCLOPS_ROOT/data/ccpd/downloads/CCPD2019.tar.xz
$TINYCYCLOPS_ROOT/data/ccpd/source
$TINYCYCLOPS_ROOT/data/ccpd/ccpd2019_index.txt
$TINYCYCLOPS_ROOT/data/ccpd/preset_images
```

Reference environment notes:

- Extracted source contains `355013` JPG images.
- The validated index contains `354998` usable source images.
- Every web Start for Preset Workload 2 draws a fresh category-balanced
  `1,000` image list from the index.
- Runtime jobs materialize only job-local symlinks under
  `runs/web_ccpd_*/input_images`.
- `data/ccpd/preset_images` is optional preview material; the web runtime uses
  the index for fresh random workloads.

Prepare or refresh the index/preview symlink set:

```bash
cd "$TINYCYCLOPS_ROOT"
./scripts/prepare_ccpd_preset.py --limit 1000 --strategy balanced --seed 20260415 --replace
```

Use `--copy` only if the source dataset will not remain mounted. The default
symlink mode avoids duplicating the large CCPD tree.

## Web Service

The service is a FastAPI parent process. Each OCR workload runs in a short-lived
child process, opens input files directly, writes progress and artifacts, then
exits so the OS can reclaim HAILO/Paddle/OpenCV native memory.

Run manually:

```bash
cd "$TINYCYCLOPS_ROOT"
./scripts/run_tinycyclops_web.sh
```

Defaults:

- `host=0.0.0.0`
- `port=18041`
- `work_block_size=20`
- `hef_batch_size=10`
- `uvicorn --workers 1`

Environment overrides:

```bash
export TINYCYCLOPS_ROOT=/path/to/tinycyclops-edge-ocr
export TINYCYCLOPS_VENV="$TINYCYCLOPS_ROOT/.venv"
export TINYCYCLOPS_WEB_HOST=0.0.0.0
export TINYCYCLOPS_WEB_PORT=18041
```

Example systemd unit:

```ini
[Unit]
Description=TinyCyclops Web Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=your-user
Group=your-user
WorkingDirectory=/path/to/tinycyclops-edge-ocr
Environment=PYTHONUNBUFFERED=1
ExecStart=/path/to/tinycyclops-edge-ocr/.venv/bin/uvicorn tinycyclops_ocr.web_app:app --host 0.0.0.0 --port 18041
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
```

The public endpoint can be handled by any reverse proxy in front of the local
TinyCyclops service. Reverse-proxy configuration is outside this repository.

## Health And Smoke Tests

Official HAILO PaddleOCR smoke test:

```bash
cd "$TINYCYCLOPS_ROOT"
./scripts/run_official_paddle_ocr.sh
```

Memory WorkBlock smoke test:

```bash
cd "$TINYCYCLOPS_ROOT"
./scripts/run_icdar_ocr_workblock_pipeline.sh --limit 10 --json
```

Web health:

```bash
curl -s http://127.0.0.1:18041/health
```

The health endpoint reports whether the ICDAR/CCPD presets, HEFs, and repo-local
logo asset are visible to the running service.

## Generated Outputs

OCR and web runs write artifacts under `runs/`. Typical outputs:

- `summary.json`
- `results.jsonl`
- `detections.csv`
- `full_text.txt`
- `run_stdout.log`
- `progress.jsonl`
- `child_stdout.log`
- `child_stderr.log`
- `workload_manifest.json` for randomized CCPD jobs

Custom uploads are temporary runtime material under `runs/web_custom_uploads`.
Do not commit `runs/`.

## Restore Checklist For Another Raspberry Pi 5

1. Clone TinyCyclops to a host-local installation path and set `TINYCYCLOPS_ROOT` to that path.
2. Install HailoRT and verify `hailortcli --version` and `hailortcli scan`.
3. Recreate `$TINYCYCLOPS_VENV` or `$TINYCYCLOPS_ROOT/.venv` with
   `--system-site-packages`.
4. Install/verify the active OCR Python modules listed above.
5. Recreate `third_party/hailo-apps` and download `ocr_det.hef` / `ocr.hef`.
6. Prepare `data/icdar2015` and `data/ccpd` as described above.
7. Install the systemd unit if this host should run the web service.
8. Check `http://127.0.0.1:18041/health` before exposing through Caddy.
