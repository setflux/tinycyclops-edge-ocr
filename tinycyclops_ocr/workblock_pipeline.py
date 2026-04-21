from __future__ import annotations

import collections
import logging
import queue
import threading
import time
import uuid
from functools import partial
from pathlib import Path

import cv2

from .metrics import PipelineMetrics
from .official_paddle_pipeline import (
    OCR_APP_DIR,
    OcrRunConfig,
    decode_ocr_detections,
    discover_images,
    emit_progress,
)
from .workblock import ImageWork, WorkBlock

from paddle_ocr import ocr_expected_counts, ocr_results_dict  # type: ignore  # noqa: E402
from paddle_ocr_utils import (  # type: ignore  # noqa: E402
    OcrCorrector,
    det_postprocess,
    resize_with_padding,
)

from hailo_apps.python.core.common.defines import (  # noqa: E402
    MAX_ASYNC_INFER_JOBS,
    MAX_INPUT_QUEUE_SIZE,
    MAX_OUTPUT_QUEUE_SIZE,
)
from hailo_apps.python.core.common.hailo_inference import HailoInfer  # noqa: E402
from hailo_apps.python.core.common.toolbox import default_preprocess  # noqa: E402


logger = logging.getLogger(__name__)


def iter_batches(items: list, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    for index in range(0, len(items), batch_size):
        yield items[index:index + batch_size]


def load_image_works(image_paths: list[Path]) -> tuple[list[object | None], list[ImageWork], dict[int, ImageWork]]:
    images: list[object | None] = []
    image_works = []
    image_work_by_id: dict[int, ImageWork] = {}

    for index, path in enumerate(image_paths):
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"Could not read image: {path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image_work = ImageWork(
            index=index,
            image_path=path,
            image_id=id(rgb),
            width=int(rgb.shape[1]),
            height=int(rgb.shape[0]),
            original_frame=rgb,
        )
        images.append(rgb)
        image_works.append(image_work)
        image_work_by_id[image_work.image_id] = image_work

    return images, image_works, image_work_by_id


def preprocess_work_blocks(
    work_blocks: list[WorkBlock],
    input_queue: queue.Queue,
    model_input_width: int,
    model_input_height: int,
    hef_batch_size: int,
    metrics: PipelineMetrics,
    stop_event: threading.Event,
) -> None:
    for work_block in work_blocks:
        if stop_event.is_set():
            break

        raw_frames = []
        for image_work in work_block.image_works:
            if image_work.original_frame is None:
                raise RuntimeError(f"ImageWork was released before preprocess: {image_work.image_path}")
            raw_frames.append(image_work.original_frame)

        started_at = time.monotonic()
        preprocessed_batch = [
            default_preprocess(frame, model_input_width, model_input_height) for frame in raw_frames
        ]
        metrics.observe("preprocess", time.monotonic() - started_at, len(raw_frames))
        metrics.increment("work_blocks_submitted")

        for raw_batch, preprocessed_sub_batch in zip(
            iter_batches(raw_frames, hef_batch_size),
            iter_batches(preprocessed_batch, hef_batch_size),
        ):
            input_queue.put((raw_batch, preprocessed_sub_batch))
            metrics.increment("hef_batches_submitted")

    input_queue.put(None)


def detector_hailo_infer_timed(
    hailo_inference: HailoInfer,
    input_queue: queue.Queue,
    output_queue: queue.Queue,
    metrics: PipelineMetrics,
    max_async_jobs: int,
    stop_event: threading.Event,
) -> None:
    pending_jobs = collections.deque()

    while True:
        next_batch = input_queue.get()
        if not next_batch:
            break

        if stop_event.is_set():
            continue

        input_batch, preprocessed_batch = next_batch
        inference_callback_fn = partial(
            detector_inference_callback_timed,
            input_batch=input_batch,
            output_queue=output_queue,
            metrics=metrics,
            submitted_at=time.monotonic(),
        )

        while len(pending_jobs) >= max_async_jobs:
            started_at = time.monotonic()
            pending_jobs.popleft().wait(10000)
            metrics.observe("detector_async_slot_wait", time.monotonic() - started_at)

        job = hailo_inference.run(preprocessed_batch, inference_callback_fn)
        pending_jobs.append(job)

    while pending_jobs:
        started_at = time.monotonic()
        pending_jobs.popleft().wait(10000)
        metrics.observe("detector_async_drain_wait", time.monotonic() - started_at)

    hailo_inference.close()
    output_queue.put(None)


def detector_inference_callback_timed(
    completion_info,
    bindings_list: list,
    input_batch: list,
    output_queue: queue.Queue,
    metrics: PipelineMetrics,
    submitted_at: float,
) -> None:
    metrics.observe("detector_hailo_async", time.monotonic() - submitted_at, len(input_batch))

    if completion_info.exception:
        logger.error("Detection inference error: %s", completion_info.exception)
        return

    for index, bindings in enumerate(bindings_list):
        result = bindings.output().get_buffer()
        output_queue.put((input_batch[index], result))


def detection_postprocess_timed(
    det_postprocess_queue: queue.Queue,
    ocr_input_queue: queue.Queue,
    output_queue: queue.Queue,
    model_height: int,
    model_width: int,
    metrics: PipelineMetrics,
    stop_event: threading.Event,
) -> None:
    while True:
        item = det_postprocess_queue.get()
        if item is None:
            break

        if stop_event.is_set():
            continue

        input_frame, result = item

        started_at = time.monotonic()
        det_pp_res, boxes = det_postprocess(result, input_frame, model_height, model_width)
        metrics.observe("detection_postprocess_crop", time.monotonic() - started_at)
        metrics.increment("crop_count", len(det_pp_res))

        frame_id = str(uuid.uuid4())
        ocr_expected_counts[frame_id] = len(det_pp_res)

        if len(det_pp_res) == 0:
            del ocr_expected_counts[frame_id]
            output_queue.put((input_frame, [], []))
            continue

        for index, cropped in enumerate(det_pp_res):
            started_at = time.monotonic()
            resized = resize_with_padding(cropped)
            metrics.observe("ocr_crop_resize", time.monotonic() - started_at)
            ocr_input_queue.put((input_frame, [resized], (frame_id, boxes[index])))

    ocr_input_queue.put(None)


def ocr_hailo_infer_timed(
    hailo_inference: HailoInfer,
    input_queue: queue.Queue,
    output_queue: queue.Queue,
    metrics: PipelineMetrics,
    max_async_jobs: int,
    stop_event: threading.Event,
) -> None:
    pending_jobs = collections.deque()

    while True:
        next_batch = input_queue.get()
        if not next_batch:
            break

        if stop_event.is_set():
            continue

        input_batch, preprocessed_batch, extra_context = next_batch
        inference_callback_fn = partial(
            ocr_inference_callback_timed,
            input_batch=input_batch,
            output_queue=output_queue,
            extra_context=extra_context,
            metrics=metrics,
            submitted_at=time.monotonic(),
        )

        while len(pending_jobs) >= max_async_jobs:
            started_at = time.monotonic()
            pending_jobs.popleft().wait(10000)
            metrics.observe("ocr_async_slot_wait", time.monotonic() - started_at)

        job = hailo_inference.run(preprocessed_batch, inference_callback_fn)
        pending_jobs.append(job)

    while pending_jobs:
        started_at = time.monotonic()
        pending_jobs.popleft().wait(10000)
        metrics.observe("ocr_async_drain_wait", time.monotonic() - started_at)

    hailo_inference.close()
    output_queue.put(None)


def ocr_inference_callback_timed(
    completion_info,
    bindings_list: list,
    input_batch,
    output_queue: queue.Queue,
    extra_context,
    metrics: PipelineMetrics,
    submitted_at: float,
) -> None:
    metrics.observe("ocr_hailo_async", time.monotonic() - submitted_at)

    if completion_info.exception:
        logger.error("OCR inference error: %s", completion_info.exception)
        return

    result = bindings_list[0].output().get_buffer()
    frame_id, box = extra_context
    output_queue.put((frame_id, input_batch, result, box))


def ocr_postprocess_timed(
    ocr_postprocess_queue: queue.Queue,
    output_queue: queue.Queue,
    metrics: PipelineMetrics,
    stop_event: threading.Event,
) -> None:
    while True:
        item = ocr_postprocess_queue.get()
        if item is None:
            break

        if stop_event.is_set():
            continue

        started_at = time.monotonic()
        frame_id, original_frame, ocr_output, denorm_box = item
        ocr_results_dict[frame_id]["results"].append(ocr_output)
        ocr_results_dict[frame_id]["boxes"].append(denorm_box)
        ocr_results_dict[frame_id]["count"] += 1
        ocr_results_dict[frame_id]["frame"] = original_frame

        expected = ocr_expected_counts.get(frame_id)
        if expected is not None and ocr_results_dict[frame_id]["count"] == expected:
            output_queue.put(
                (
                    ocr_results_dict[frame_id]["frame"],
                    ocr_results_dict[frame_id]["results"],
                    ocr_results_dict[frame_id]["boxes"],
                )
            )
            del ocr_results_dict[frame_id]
            del ocr_expected_counts[frame_id]

        metrics.observe("ocr_postprocess_group", time.monotonic() - started_at)

    output_queue.put(None)


def run_memory_workblock_ocr(config: OcrRunConfig) -> tuple[list[dict], dict]:
    image_paths = discover_images(config.input_path, config.limit)
    metrics = PipelineMetrics()
    emit_progress(
        config.progress_callback,
        "discovered",
        input_path=str(config.input_path),
        image_count=len(image_paths),
        limit=config.limit,
        hef_batch_size=config.batch_size,
        work_block_size=config.work_block_size,
    )

    started_at = time.monotonic()
    images, image_works, image_work_by_id = load_image_works(image_paths)
    image_load_seconds = time.monotonic() - started_at
    metrics.observe("image_load", image_load_seconds, len(image_works))
    work_blocks = WorkBlock.chunked(image_works, config.work_block_size)
    emit_progress(
        config.progress_callback,
        "images_loaded",
        image_count=len(image_works),
        work_block_count=len(work_blocks),
        image_load_seconds=image_load_seconds,
    )

    ocr_results_dict.clear()
    ocr_expected_counts.clear()

    stop_event = threading.Event()
    det_input_queue: queue.Queue = queue.Queue(maxsize=MAX_INPUT_QUEUE_SIZE)
    ocr_input_queue: queue.Queue = queue.Queue(maxsize=MAX_INPUT_QUEUE_SIZE)
    det_postprocess_queue: queue.Queue = queue.Queue(maxsize=MAX_INPUT_QUEUE_SIZE)
    ocr_postprocess_queue: queue.Queue = queue.Queue(maxsize=MAX_INPUT_QUEUE_SIZE)
    output_queue: queue.Queue = queue.Queue(maxsize=MAX_OUTPUT_QUEUE_SIZE)

    hef_load_started_at = time.monotonic()
    detector_hailo_inference = HailoInfer(str(config.det_hef), config.batch_size)
    ocr_hailo_inference = HailoInfer(str(config.ocr_hef), config.batch_size, priority=1)
    model_height, model_width, _ = detector_hailo_inference.get_input_shape()
    hef_load_seconds = time.monotonic() - hef_load_started_at
    metrics.observe("hef_load", hef_load_seconds, 2)
    detector_max_async_jobs = max(1, MAX_ASYNC_INFER_JOBS // config.batch_size)
    ocr_max_async_jobs = MAX_ASYNC_INFER_JOBS
    emit_progress(
        config.progress_callback,
        "engine_ready",
        hef_load_seconds=hef_load_seconds,
        detector_input_shape={
            "height": int(model_height),
            "width": int(model_width),
            "channels": 3,
        },
        detector_max_async_jobs=detector_max_async_jobs,
        ocr_max_async_jobs=ocr_max_async_jobs,
    )

    ocr_corrector = None
    if config.use_corrector:
        dictionary_path = OCR_APP_DIR / "frequency_dictionary_en_82_765.txt"
        ocr_corrector = OcrCorrector(dictionary_path=str(dictionary_path))

    threads = [
        threading.Thread(
            target=preprocess_work_blocks,
            args=(
                work_blocks,
                det_input_queue,
                model_width,
                model_height,
                config.batch_size,
                metrics,
                stop_event,
            ),
            name="tinycyclops-workblock-preprocess",
        ),
        threading.Thread(
            target=detector_hailo_infer_timed,
            args=(
                detector_hailo_inference,
                det_input_queue,
                det_postprocess_queue,
                metrics,
                detector_max_async_jobs,
                stop_event,
            ),
            name="tinycyclops-workblock-detector-infer",
        ),
        threading.Thread(
            target=detection_postprocess_timed,
            args=(
                det_postprocess_queue,
                ocr_input_queue,
                output_queue,
                model_height,
                model_width,
                metrics,
                stop_event,
            ),
            name="tinycyclops-workblock-crop-broker",
        ),
        threading.Thread(
            target=ocr_hailo_infer_timed,
            args=(
                ocr_hailo_inference,
                ocr_input_queue,
                ocr_postprocess_queue,
                metrics,
                ocr_max_async_jobs,
                stop_event,
            ),
            name="tinycyclops-workblock-ocr-infer",
        ),
        threading.Thread(
            target=ocr_postprocess_timed,
            args=(ocr_postprocess_queue, output_queue, metrics, stop_event),
            name="tinycyclops-workblock-ocr-postprocess",
        ),
    ]

    released_image_count = 0
    total_detection_count = 0
    nonempty_image_count = 0
    run_started_at = time.monotonic()
    emit_progress(
        config.progress_callback,
        "started",
        image_count=len(image_works),
        work_block_count=len(work_blocks),
        hef_batch_size=config.batch_size,
        work_block_size=config.work_block_size,
    )
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
                image_work = image_work_by_id.pop(id(original_frame), None)
                if image_work is None:
                    raise RuntimeError("Could not map OCR result back to its source ImageWork.")

                detections = decode_ocr_detections(raw_ocr_results, boxes, ocr_corrector)
                total_detection_count += len(detections)
                image_text = " ".join(
                    detection["text"].strip()
                    for detection in detections
                    if detection["text"].strip()
                )
                if image_text.strip():
                    nonempty_image_count += 1

                results_by_path[image_work.image_path] = {
                    "image": str(image_work.image_path),
                    "image_name": image_work.image_path.name,
                    "width": image_work.width,
                    "height": image_work.height,
                    "text": image_text,
                    "detections": detections,
                }

                images[image_work.index] = None
                image_work.release_original()
                released_image_count += 1
                metrics.increment("image_work_released")
                elapsed_seconds = time.monotonic() - run_started_at
                emit_progress(
                    config.progress_callback,
                    "image_completed",
                    image=str(image_work.image_path),
                    image_name=image_work.image_path.name,
                    width=image_work.width,
                    height=image_work.height,
                    processed_image_count=released_image_count,
                    image_count=len(image_works),
                    detection_count=len(detections),
                    total_detection_count=total_detection_count,
                    nonempty_image_count=nonempty_image_count,
                    fps=released_image_count / elapsed_seconds if elapsed_seconds else 0.0,
                    elapsed_seconds=elapsed_seconds,
                    text_preview=image_text[:160],
                )
            finally:
                output_queue.task_done()
    finally:
        stop_event.set()
        for thread in threads:
            thread.join()

    elapsed_seconds = time.monotonic() - run_started_at
    remaining_resident_images = sum(1 for image in images if image is not None)
    for index, image in enumerate(images):
        if image is not None:
            images[index] = None
            metrics.increment("image_work_final_cleanup")

    ordered_results = [
        results_by_path.get(
            image_work.image_path,
            {
                "image": str(image_work.image_path),
                "image_name": image_work.image_path.name,
                "width": image_work.width,
                "height": image_work.height,
                "text": "",
                "detections": [],
                "error": "no_result",
            },
        )
        for image_work in image_works
    ]

    summary = {
        "pipeline": "memory_workblock",
        "input_path": str(config.input_path),
        "det_hef": str(config.det_hef),
        "ocr_hef": str(config.ocr_hef),
        "image_count": len(ordered_results),
        "hef_batch_size": config.batch_size,
        "work_block_size": config.work_block_size,
        "work_block_count": len(work_blocks),
        "detector_max_async_jobs": detector_max_async_jobs,
        "ocr_max_async_jobs": ocr_max_async_jobs,
        "detection_count": sum(len(result["detections"]) for result in ordered_results),
        "nonempty_image_count": sum(1 for result in ordered_results if result["text"].strip()),
        "elapsed_seconds": elapsed_seconds,
        "fps": len(ordered_results) / elapsed_seconds if elapsed_seconds else 0.0,
        "use_corrector": config.use_corrector,
        "image_work_released_count": released_image_count,
        "resident_image_count_after_run": remaining_resident_images,
        "metrics": metrics.snapshot(),
    }
    emit_progress(
        config.progress_callback,
        "completed",
        image_count=len(ordered_results),
        detection_count=summary["detection_count"],
        nonempty_image_count=summary["nonempty_image_count"],
        elapsed_seconds=summary["elapsed_seconds"],
        fps=summary["fps"],
        image_work_released_count=released_image_count,
        resident_image_count_after_run=remaining_resident_images,
    )

    return ordered_results, summary
