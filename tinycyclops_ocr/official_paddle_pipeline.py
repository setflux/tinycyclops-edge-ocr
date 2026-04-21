from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2

from .runtime import (
    DEFAULT_DET_HEF,
    DEFAULT_ICDAR_IMAGES,
    DEFAULT_OCR_HEF,
    HAILO_APPS_ROOT,
    IMAGE_EXTENSIONS,
    OCR_APP_DIR,
    PROJECT_ROOT,
    OcrRunArtifacts,
    natural_sort_key,
    write_result_artifacts,
)

for import_path in (str(OCR_APP_DIR), str(HAILO_APPS_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from paddle_ocr import (  # type: ignore  # noqa: E402
    detection_postprocess,
    detector_hailo_infer,
    ocr_expected_counts,
    ocr_hailo_infer,
    ocr_postprocess,
    ocr_results_dict,
)
from paddle_ocr_utils import OcrCorrector, ocr_eval_postprocess  # type: ignore  # noqa: E402

from hailo_apps.python.core.common.defines import (  # noqa: E402
    MAX_INPUT_QUEUE_SIZE,
    MAX_OUTPUT_QUEUE_SIZE,
)
from hailo_apps.python.core.common.hailo_inference import HailoInfer  # noqa: E402
from hailo_apps.python.core.common.toolbox import (  # noqa: E402
    InputContext,
    InputType,
    preprocess,
)


ProgressCallback = Callable[[dict], None]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OcrRunConfig:
    input_path: Path = DEFAULT_ICDAR_IMAGES
    output_dir: Path = PROJECT_ROOT / "runs" / "ocr_text"
    det_hef: Path = DEFAULT_DET_HEF
    ocr_hef: Path = DEFAULT_OCR_HEF
    limit: int | None = None
    batch_size: int = 1
    work_block_size: int = 1
    use_corrector: bool = False
    progress_callback: ProgressCallback | None = None


def discover_images(input_path: Path, limit: int | None = None) -> list[Path]:
    input_path = input_path.expanduser().resolve()

    if input_path.is_file() and input_path.suffix.lower() in IMAGE_EXTENSIONS:
        images = [input_path]
    elif input_path.is_dir():
        images = sorted(
            (
                path
                for path in input_path.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=natural_sort_key,
        )
    else:
        raise FileNotFoundError(f"No supported image input found: {input_path}")

    if limit is not None:
        images = images[:limit]

    if not images:
        raise ValueError(f"No images to process: {input_path}")

    return images


def emit_progress(progress_callback: ProgressCallback | None, event: str, **payload) -> None:
    if progress_callback is None:
        return

    try:
        progress_callback({"event": event, **payload})
    except Exception:
        logger.exception("OCR progress callback failed")


def load_rgb_images(image_paths: Iterable[Path]) -> tuple[list[object], dict[int, Path]]:
    images = []
    image_path_by_id: dict[int, Path] = {}

    for path in image_paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"Could not read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        images.append(rgb)
        image_path_by_id[id(rgb)] = path

    return images, image_path_by_id


def decode_ocr_detections(raw_ocr_results, boxes, ocr_corrector: OcrCorrector | None) -> list[dict]:
    detections = []

    for index, (raw_result, box) in enumerate(zip(raw_ocr_results, boxes)):
        text, raw_confidence = ocr_eval_postprocess(raw_result)[0]
        if ocr_corrector is not None and text.strip():
            text = ocr_corrector.correct_text(text)

        raw_confidence = float(raw_confidence)
        confidence = raw_confidence / 255.0 if raw_confidence > 1.0 else raw_confidence
        confidence = max(0.0, min(1.0, confidence))

        x, y, w, h = [int(v) for v in box]
        detections.append(
            {
                "index": index,
                "box": {"x": x, "y": y, "w": w, "h": h},
                "text": text,
                "confidence": confidence,
                "raw_confidence": raw_confidence,
            }
        )

    return detections


def run_official_paddle_ocr(config: OcrRunConfig) -> tuple[list[dict], dict]:
    image_paths = discover_images(config.input_path, config.limit)
    images, image_path_by_id = load_rgb_images(image_paths)

    ocr_results_dict.clear()
    ocr_expected_counts.clear()

    input_context = InputContext(
        input_src=str(config.input_path),
        batch_size=config.batch_size,
    )
    input_context.input_type = InputType.IMAGES
    input_context.images = images

    stop_event = threading.Event()
    det_input_queue: queue.Queue = queue.Queue(maxsize=MAX_INPUT_QUEUE_SIZE)
    ocr_input_queue: queue.Queue = queue.Queue(maxsize=MAX_INPUT_QUEUE_SIZE)
    det_postprocess_queue: queue.Queue = queue.Queue(maxsize=MAX_INPUT_QUEUE_SIZE)
    ocr_postprocess_queue: queue.Queue = queue.Queue(maxsize=MAX_INPUT_QUEUE_SIZE)
    output_queue: queue.Queue = queue.Queue(maxsize=MAX_OUTPUT_QUEUE_SIZE)

    detector_hailo_inference = HailoInfer(str(config.det_hef), config.batch_size)
    ocr_hailo_inference = HailoInfer(str(config.ocr_hef), config.batch_size, priority=1)
    model_height, model_width, _ = detector_hailo_inference.get_input_shape()

    ocr_corrector = None
    if config.use_corrector:
        dictionary_path = OCR_APP_DIR / "frequency_dictionary_en_82_765.txt"
        ocr_corrector = OcrCorrector(dictionary_path=str(dictionary_path))

    threads = [
        threading.Thread(
            target=preprocess,
            args=(input_context, det_input_queue, model_width, model_height, None, stop_event),
            name="tinycyclops-preprocess",
        ),
        threading.Thread(
            target=detector_hailo_infer,
            args=(detector_hailo_inference, det_input_queue, det_postprocess_queue, stop_event),
            name="tinycyclops-detector-infer",
        ),
        threading.Thread(
            target=detection_postprocess,
            args=(
                det_postprocess_queue,
                ocr_input_queue,
                output_queue,
                model_height,
                model_width,
                stop_event,
            ),
            name="tinycyclops-detection-postprocess",
        ),
        threading.Thread(
            target=ocr_hailo_infer,
            args=(ocr_hailo_inference, ocr_input_queue, ocr_postprocess_queue, stop_event),
            name="tinycyclops-ocr-infer",
        ),
        threading.Thread(
            target=ocr_postprocess,
            args=(ocr_postprocess_queue, output_queue, stop_event),
            name="tinycyclops-ocr-postprocess",
        ),
    ]

    started_at = time.monotonic()
    for thread in threads:
        thread.start()

    results_by_path: dict[Path, dict] = {}
    try:
        while True:
            item = output_queue.get()
            try:
                if item is None:
                    break

                original_frame, raw_ocr_results, boxes = item
                image_path = image_path_by_id.get(id(original_frame))
                if image_path is None:
                    raise RuntimeError("Could not map OCR result back to its source image path.")

                detections = decode_ocr_detections(raw_ocr_results, boxes, ocr_corrector)
                image_text = " ".join(
                    detection["text"].strip()
                    for detection in detections
                    if detection["text"].strip()
                )

                results_by_path[image_path] = {
                    "image": str(image_path),
                    "image_name": image_path.name,
                    "width": int(original_frame.shape[1]),
                    "height": int(original_frame.shape[0]),
                    "text": image_text,
                    "detections": detections,
                }
            finally:
                output_queue.task_done()
    finally:
        stop_event.set()
        for thread in threads:
            thread.join()

    elapsed_seconds = time.monotonic() - started_at
    ordered_results = [
        results_by_path.get(
            image_path,
            {
                "image": str(image_path),
                "image_name": image_path.name,
                "text": "",
                "detections": [],
                "error": "no_result",
            },
        )
        for image_path in image_paths
    ]

    summary = {
        "input_path": str(config.input_path),
        "det_hef": str(config.det_hef),
        "ocr_hef": str(config.ocr_hef),
        "image_count": len(ordered_results),
        "hef_batch_size": config.batch_size,
        "detection_count": sum(len(result["detections"]) for result in ordered_results),
        "nonempty_image_count": sum(1 for result in ordered_results if result["text"].strip()),
        "elapsed_seconds": elapsed_seconds,
        "fps": len(ordered_results) / elapsed_seconds if elapsed_seconds else 0.0,
        "use_corrector": config.use_corrector,
    }

    return ordered_results, summary
