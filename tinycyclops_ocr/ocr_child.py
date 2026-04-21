from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one TinyCyclops OCR job in an isolated process.")
    parser.add_argument("--job-spec", type=Path, required=True, help="JSON job specification path.")
    return parser.parse_args()


def read_job_spec(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_progress(progress_jsonl: Path, event: dict[str, Any]) -> None:
    with progress_jsonl.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(event, ensure_ascii=False) + "\n")
        fp.flush()


def main() -> int:
    args = parse_args()
    spec = read_job_spec(args.job_spec)
    output_dir = Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_jsonl = Path(spec["progress_jsonl"])
    run_stdout_log = Path(spec["run_stdout_log"])
    run_stdout_log.parent.mkdir(parents=True, exist_ok=True)
    progress_jsonl.parent.mkdir(parents=True, exist_ok=True)
    progress_jsonl.touch(exist_ok=True)

    try:
        # Heavy OCR/HAILO imports stay inside this short-lived child process.
        from .official_paddle_pipeline import OcrRunConfig, write_result_artifacts
        from .workblock_pipeline import run_memory_workblock_ocr

        config = OcrRunConfig(
            input_path=Path(spec["input_path"]),
            output_dir=output_dir,
            det_hef=Path(spec["det_hef"]),
            ocr_hef=Path(spec["ocr_hef"]),
            limit=spec.get("limit"),
            batch_size=int(spec["hef_batch_size"]),
            work_block_size=int(spec["work_block_size"]),
            use_corrector=bool(spec.get("use_corrector", False)),
            progress_callback=lambda event: write_progress(progress_jsonl, event),
        )

        with run_stdout_log.open("w", encoding="utf-8") as log_fp:
            with contextlib.redirect_stdout(log_fp):
                results, summary = run_memory_workblock_ocr(config)
        write_result_artifacts(results, summary, output_dir, run_stdout_log)
        return 0
    except Exception as exc:
        error_payload = {
            "event": "child_failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        with contextlib.suppress(Exception):
            write_progress(progress_jsonl, error_payload)
        print(error_payload["traceback"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
