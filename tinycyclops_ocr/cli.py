from __future__ import annotations

import argparse
import contextlib
import json
from datetime import datetime
from pathlib import Path

from .official_paddle_pipeline import (
    DEFAULT_DET_HEF,
    DEFAULT_ICDAR_IMAGES,
    DEFAULT_OCR_HEF,
    PROJECT_ROOT,
    OcrRunConfig,
    run_official_paddle_ocr,
    write_result_artifacts,
)
from .workblock_pipeline import run_memory_workblock_ocr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TinyCyclops text OCR over an image set using HAILO's official PaddleOCR HEFs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_ICDAR_IMAGES,
        help="Image file or directory. Defaults to TinyCyclops ICDAR 2015 test_images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summary.json, results.jsonl, detections.csv, and full_text.txt.",
    )
    parser.add_argument("--det-hef", type=Path, default=DEFAULT_DET_HEF)
    parser.add_argument("--ocr-hef", type=Path, default=DEFAULT_OCR_HEF)
    parser.add_argument(
        "--hef-batch-size",
        type=int,
        default=1,
        help="Number of inputs sent in each HEF inference job. Defaults to 1.",
    )
    parser.add_argument(
        "--work-block-size",
        type=int,
        default=1,
        help="Number of ImageWork items grouped by the memory-workblock broker. Defaults to 1.",
    )
    parser.add_argument(
        "--pipeline",
        choices=("official", "memory-workblock"),
        default="official",
        help="Pipeline implementation to run. Defaults to the existing official wrapper.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N images.")
    parser.add_argument("--use-corrector", action="store_true", help="Enable SymSpell text correction.")
    parser.add_argument(
        "--run-stdout-log",
        type=Path,
        default=None,
        help="Capture stdout from the official HAILO pipeline. Defaults to OUTPUT_DIR/run_stdout.log.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable run metadata.")
    args = parser.parse_args()
    if args.hef_batch_size < 1:
        parser.error("--hef-batch-size must be greater than zero")
    if args.work_block_size < 1:
        parser.error("--work-block-size must be greater than zero")
    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_prefix = "icdar2015_ocr_workblock" if args.pipeline == "memory-workblock" else "icdar2015_ocr_text"
        output_dir = PROJECT_ROOT / "runs" / f"{run_prefix}_{stamp}"

    config = OcrRunConfig(
        input_path=args.input,
        output_dir=output_dir,
        det_hef=args.det_hef,
        ocr_hef=args.ocr_hef,
        limit=args.limit,
        batch_size=args.hef_batch_size,
        work_block_size=args.work_block_size,
        use_corrector=args.use_corrector,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    run_stdout_log = args.run_stdout_log or output_dir / "run_stdout.log"
    run_stdout_log.parent.mkdir(parents=True, exist_ok=True)

    with run_stdout_log.open("w", encoding="utf-8") as log_fp:
        with contextlib.redirect_stdout(log_fp):
            if args.pipeline == "memory-workblock":
                results, summary = run_memory_workblock_ocr(config)
            else:
                results, summary = run_official_paddle_ocr(config)

    artifacts = write_result_artifacts(results, summary, output_dir, run_stdout_log)

    payload = {
        "status": "ok",
        "summary": summary,
        "artifacts": {
            "output_dir": str(artifacts.output_dir),
            "summary_json": str(artifacts.summary_json),
            "results_jsonl": str(artifacts.results_jsonl),
            "detections_csv": str(artifacts.detections_csv),
            "full_text_txt": str(artifacts.full_text_txt),
            "run_stdout_log": str(run_stdout_log),
        },
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"status=ok")
        print(f"output_dir={artifacts.output_dir}")
        print(f"image_count={summary['image_count']}")
        print(f"detection_count={summary['detection_count']}")
        print(f"nonempty_image_count={summary['nonempty_image_count']}")
        print(f"elapsed_seconds={summary['elapsed_seconds']:.3f}")
        print(f"fps={summary['fps']:.3f}")


if __name__ == "__main__":
    main()
