# TinyCyclops Independent Installation

This guide is for a public-repo user setting up TinyCyclops on their own
Raspberry Pi + Hailo host.

TinyCyclops source code is portable, but real OCR inference depends on external
runtime pieces that are not stored in this repository: HailoRT, a visible Hailo
device, the official `hailo-apps` checkout, downloaded OCR HEF assets, and local
datasets.

## 1. Target Environment

Tested reference environment:

- Board: Raspberry Pi 5 class host.
- OS family: 64-bit Debian / Raspberry Pi OS style Linux.
- Accelerator: Hailo-8 device visible to HailoRT.
- HailoRT: `4.23.0`.
- Python: `3.13.5` in a virtual environment created with `--system-site-packages`.
- Hailo Apps revision: `891ce701c2ebe239a5d277759eb75a30f76678a9`
  (`git describe`: `26.03.1`).

Other Debian-like Raspberry Pi setups may work, but the versions above are the
known-good baseline.

## 2. System Prerequisites

Install basic host tools:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip wget curl unzip xz-utils
```

Install HailoRT and the Hailo device driver using Hailo's official installation
path for your device and OS. This repository does not redistribute HailoRT,
driver packages, firmware, or Hailo Python bindings.

After installing HailoRT, verify the device before continuing:

```bash
hailortcli --version
hailortcli scan
```

Expected result:

- `hailortcli --version` prints a HailoRT version.
- `hailortcli scan` lists a Hailo device.

If the device does not appear here, fix HailoRT/driver/PCIe setup before working
on TinyCyclops.

## 3. Clone TinyCyclops

```bash
git clone https://github.com/YOUR_ORG/tinycyclops-edge-ocr.git
cd tinycyclops-edge-ocr
export TINYCYCLOPS_ROOT="$PWD"
```

If you install the repository somewhere else later, set `TINYCYCLOPS_ROOT` to
that directory.

## 4. Create The Python Environment

The venv should be created with `--system-site-packages` so Python can see
HailoRT / TAPPAS bindings installed by the system-level Hailo stack.

```bash
cd "$TINYCYCLOPS_ROOT"
python3 -m venv --system-site-packages .venv
export TINYCYCLOPS_VENV="$TINYCYCLOPS_ROOT/.venv"
. "$TINYCYCLOPS_VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements-web.txt
```

Check that the venv can see the Hailo Python stack:

```bash
python - <<'PY'
import hailo_platform
print("hailo_platform import ok")
PY
```

If this import fails, the venv probably was not created with
`--system-site-packages`, or the Hailo Python bindings are not installed in the
host Python environment.

## 5. Install Hailo Apps And OCR Assets

TinyCyclops imports the official PaddleOCR app from `third_party/hailo-apps`.
The directory is intentionally git-ignored.

```bash
cd "$TINYCYCLOPS_ROOT"
mkdir -p third_party
git clone https://github.com/hailo-ai/hailo-apps.git third_party/hailo-apps
cd third_party/hailo-apps
git checkout 891ce701c2ebe239a5d277759eb75a30f76678a9
```

Install the official PaddleOCR Python dependencies:

```bash
cd "$TINYCYCLOPS_ROOT/third_party/hailo-apps/hailo_apps/python/standalone_apps/paddle_ocr"
python -m pip install -r requirements.txt
```

Download the Hailo-8 OCR HEFs and sample images:

```bash
cd "$TINYCYCLOPS_ROOT/third_party/hailo-apps/hailo_apps/python/standalone_apps/paddle_ocr"
bash ./download_resources.sh --arch 8
```

Expected files:

```text
$TINYCYCLOPS_ROOT/third_party/hailo-apps/hailo_apps/python/standalone_apps/paddle_ocr/ocr_det.hef
$TINYCYCLOPS_ROOT/third_party/hailo-apps/hailo_apps/python/standalone_apps/paddle_ocr/ocr.hef
$TINYCYCLOPS_ROOT/third_party/hailo-apps/hailo_apps/python/standalone_apps/paddle_ocr/ocr_img1.png
$TINYCYCLOPS_ROOT/third_party/hailo-apps/hailo_apps/python/standalone_apps/paddle_ocr/ocr_img2.png
```

## 6. Run A Hailo OCR Smoke Test

```bash
cd "$TINYCYCLOPS_ROOT"
./scripts/run_official_paddle_ocr.sh
```

Expected result:

- The command prints an `output_dir=...` line.
- The output directory contains an annotated OCR output image.

If this fails, verify:

- `hailortcli scan` sees the device.
- `ocr_det.hef` and `ocr.hef` exist.
- The active Python is `$TINYCYCLOPS_VENV/bin/python`.
- `hailo_platform`, `cv2`, `numpy`, `paddlepaddle`, `shapely`,
  `pyclipper`, `symspellpy`, `python-dotenv`, and `PyYAML` import successfully.

## 7. Prepare Datasets

TinyCyclops does not commit datasets. The default layout is:

```text
$TINYCYCLOPS_ROOT/data/icdar2015/test_images
$TINYCYCLOPS_ROOT/data/icdar2015/gt
$TINYCYCLOPS_ROOT/data/ccpd/source
$TINYCYCLOPS_ROOT/data/ccpd/ccpd2019_index.txt
$TINYCYCLOPS_ROOT/data/ccpd/preset_images
```

ICDAR 2015:

- Download the Challenge 4 test image archive from the RRC downloads page.
- Download the Challenge 4 Task 1 ground-truth archive.
- Extract images to `data/icdar2015/test_images`.
- Extract ground truth files to `data/icdar2015/gt`.

CCPD2019:

- Download `CCPD2019.tar.xz` from the Zenodo CCPD record.
- Extract it under `data/ccpd/source`.
- Build the local image index and optional preview preset:

```bash
cd "$TINYCYCLOPS_ROOT"
./scripts/prepare_ccpd_preset.py --limit 1000 --strategy balanced --replace
```

For an initial web smoke test, ICDAR is enough. CCPD is only needed for the
license-plate preset.

## 8. Run TinyCyclops

Run a small CLI OCR workload:

```bash
cd "$TINYCYCLOPS_ROOT"
./scripts/run_icdar_ocr_workblock_pipeline.sh --limit 10 --json
```

Run the web service:

```bash
cd "$TINYCYCLOPS_ROOT"
./scripts/run_tinycyclops_web.sh
```

Open or probe:

```bash
curl -s http://127.0.0.1:18041/health
```

The health response should report whether HEFs and dataset presets are visible.

## 9. Optional Environment Variables

```bash
export TINYCYCLOPS_ROOT=/path/to/tinycyclops-edge-ocr
export TINYCYCLOPS_VENV="$TINYCYCLOPS_ROOT/.venv"
export TINYCYCLOPS_WEB_HOST=0.0.0.0
export TINYCYCLOPS_WEB_PORT=18041
export TINY_CYCLOPS_OCR_DET_HEF=/path/to/ocr_det.hef
export TINY_CYCLOPS_OCR_HEF=/path/to/ocr.hef
```

`TINYCYCLOPS_ROOT` controls where scripts look for the repo. If it is unset,
scripts infer the repo root from their own location.

`TINYCYCLOPS_VENV` controls which Python runtime scripts use. If it is unset,
scripts default to `$TINYCYCLOPS_ROOT/.venv`.

## 10. Troubleshooting

- `hailortcli scan` shows no device: fix HailoRT, driver, PCIe, udev, or reboot
  before debugging TinyCyclops.
- `import hailo_platform` fails: recreate the venv with `--system-site-packages`
  after the Hailo Python bindings are installed.
- `ocr_det.hef` or `ocr.hef` missing: rerun `download_resources.sh --arch 8` in
  the PaddleOCR app directory.
- ICDAR preset is unhealthy: check that `data/icdar2015/test_images` contains
  the extracted test JPG files.
- CCPD preset is unhealthy: check `data/ccpd/source` and rebuild the index with
  `scripts/prepare_ccpd_preset.py`.
- Web uploads fail: ensure `python-multipart` is installed in
  `$TINYCYCLOPS_VENV`.
