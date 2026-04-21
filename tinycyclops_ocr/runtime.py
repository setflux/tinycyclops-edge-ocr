from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAILO_APPS_ROOT = PROJECT_ROOT / "third_party" / "hailo-apps"
OCR_APP_DIR = HAILO_APPS_ROOT / "hailo_apps" / "python" / "standalone_apps" / "paddle_ocr"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_DET_HEF = OCR_APP_DIR / "ocr_det.hef"
DEFAULT_OCR_HEF = OCR_APP_DIR / "ocr.hef"
DEFAULT_ICDAR_IMAGES = PROJECT_ROOT / "data" / "icdar2015" / "test_images"
DEFAULT_CCPD_IMAGES = PROJECT_ROOT / "data" / "ccpd" / "preset_images"


@dataclass(frozen=True)
class OcrRunArtifacts:
    output_dir: Path
    summary_json: Path
    results_jsonl: Path
    detections_csv: Path
    full_text_txt: Path


def natural_sort_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def display_path(path: Path) -> str:
    resolved = path.expanduser().resolve(strict=False)
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return resolved.name


def artifact_paths(output_dir: Path) -> OcrRunArtifacts:
    return OcrRunArtifacts(
        output_dir=output_dir,
        summary_json=output_dir / "summary.json",
        results_jsonl=output_dir / "results.jsonl",
        detections_csv=output_dir / "detections.csv",
        full_text_txt=output_dir / "full_text.txt",
    )


def read_results_jsonl(results_jsonl: Path) -> list[dict]:
    with results_jsonl.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def read_summary_json(summary_json: Path) -> dict:
    return json.loads(summary_json.read_text(encoding="utf-8"))


def write_result_artifacts(
    results: list[dict],
    summary: dict,
    output_dir: Path,
    run_stdout_log: Path | None = None,
) -> OcrRunArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = artifact_paths(output_dir)

    summary = dict(summary)
    summary["artifacts"] = {
        "summary_json": str(artifacts.summary_json),
        "results_jsonl": str(artifacts.results_jsonl),
        "detections_csv": str(artifacts.detections_csv),
        "full_text_txt": str(artifacts.full_text_txt),
    }
    if run_stdout_log is not None:
        summary["artifacts"]["run_stdout_log"] = str(run_stdout_log)

    artifacts.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with artifacts.results_jsonl.open("w", encoding="utf-8") as fp:
        for result in results:
            fp.write(json.dumps(result, ensure_ascii=False) + "\n")

    with artifacts.detections_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "image",
                "image_name",
                "detection_index",
                "x",
                "y",
                "w",
                "h",
                "confidence",
                "raw_confidence",
                "text",
            ],
        )
        writer.writeheader()
        for result in results:
            for detection in result["detections"]:
                box = detection["box"]
                writer.writerow(
                    {
                        "image": result["image"],
                        "image_name": result["image_name"],
                        "detection_index": detection["index"],
                        "x": box["x"],
                        "y": box["y"],
                        "w": box["w"],
                        "h": box["h"],
                        "confidence": f"{detection['confidence']:.6f}",
                        "raw_confidence": f"{detection['raw_confidence']:.6f}",
                        "text": detection["text"],
                    }
                )

    full_text_lines = []
    for result in results:
        text = " ".join(result["text"].split())
        full_text_lines.append(f"{result['image_name']}\t{text}")

    artifacts.full_text_txt.write_text("\n".join(full_text_lines) + "\n", encoding="utf-8")
    return artifacts
