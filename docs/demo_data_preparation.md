# Demo Data Preparation

TinyCyclops does not commit datasets. Prepare datasets locally under
`$TINYCYCLOPS_ROOT/data`.

Datasets are not required for the official Hailo PaddleOCR smoke test. They are
required only when running TinyCyclops preset workloads through the CLI or web
UI. Use ICDAR 2015 first for a small public OCR preset; CCPD2019 is optional and
large.

## ICDAR 2015 Challenge 4 Test Set

Source:

- `https://rrc.cvc.uab.es/?ch=4&com=downloads`
- `https://rrc.cvc.uab.es/downloads/ch4_test_images.zip`
- `https://rrc.cvc.uab.es/downloads/Challenge4_Test_Task1_GT.zip`

Local path:

```text
$TINYCYCLOPS_ROOT/data/icdar2015
```

Downloaded archives:

```text
ch4_test_images.zip
Challenge4_Test_Task1_GT.zip
```

Extracted paths:

```text
test_images/
gt/
```

Verified counts:

```text
test_images: 500 jpg files
gt: 500 gt_img_*.txt files
```

If TLS certificate validation fails when downloading from the RRC site, verify
the archive URL, HTTP status, and expected content length before using any
insecure download fallback.

## CCPD License Plate Preset

Purpose:

- Reserved for TinyCyclops `Preset Workload 2`.
- Intended as a runtime-randomized 1,000-image impression set for license plate OCR experiments.
- Uses CCPD as a public license plate image source. TinyCyclops OCR is currently
  alphanumeric-focused, so Chinese characters may be missed or misread.

Source:

- Zenodo record: `https://zenodo.org/records/15647076`
- DOI: `10.5281/zenodo.15647076`
- Dataset title: `CCPD (Chinese City Parking Dataset) for "Research on license plate recognition based on graphically supervised signal-assisted training"`
- License: `Creative Commons Attribution 4.0 International`
- Selected archive: `CCPD2019.tar.xz`
- Expected archive size: `13164924944` bytes
- Expected MD5: `0dfbca0e6fcb7cb8ea720b0eae94c735`
- Verify the archive size and MD5 after download before extracting.

Expected local path:

```text
$TINYCYCLOPS_ROOT/data/ccpd/preset_images
```

Local archive path:

```text
$TINYCYCLOPS_ROOT/data/ccpd/downloads/CCPD2019.tar.xz
```

Local source extraction path:

```text
$TINYCYCLOPS_ROOT/data/ccpd/source
```

Preparation command:

```bash
cd "$TINYCYCLOPS_ROOT"
./scripts/prepare_ccpd_preset.py --limit 1000 --strategy balanced --seed 20260415 --replace
```

Notes:

- `data/` is git-ignored and must not be committed.
- Extracted source contains `355013` JPG images.
- The validated TinyCyclops CCPD index contains `354998` usable source images.
  The remaining `15` files were filtered out because their file extension and
  image magic/header did not match TinyCyclops' supported image validation
  rules. The source files are not deleted; they are excluded from the generated
  index and therefore from randomized workloads.
- CCPD source filtering is implemented in `scripts/prepare_ccpd_preset.py`
  through `is_supported_image_file()` / `has_supported_image_magic()`. Runtime
  preset validation is also enforced in `tinycyclops_ocr/dataset_registry.py`
  before web jobs sample CCPD images.
- Preset Workload 2 randomizes a fresh `1,000` image symlink workload under each
  `runs/web_ccpd_*/input_images` directory when the web Start button is pressed.
- `data/ccpd/preset_images` may still be used as an optional prepared preview set.
- Preset directories may contain regular files or symbolic links. The web image
  endpoints intentionally support both.
- Dataset preset metadata is centralized in `tinycyclops_ocr/dataset_registry.py`.
- The CCPD preparation script builds or reuses `data/ccpd/ccpd2019_index.txt`.
- The default CCPD selection strategy is category-balanced random sampling.
- Use `--seed` to make a random 1,000-image preset reproducible.
- Runtime web jobs use an automatically generated seed so every Start produces a
  new category-balanced image list.
- The preparation script creates symlinks by default to avoid duplicating a large CCPD source tree.
- Use `--copy` only when the source dataset will not remain mounted.
- Public attribution should cite the Zenodo record and CC-BY-4.0 license when
  this preset is exposed on the web service.
