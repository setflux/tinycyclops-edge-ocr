from __future__ import annotations

from functools import lru_cache
import json
import re
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .dataset_registry import count_images, get_dataset_preset, list_dataset_presets, preset_health
from .runtime import (
    DEFAULT_DET_HEF,
    DEFAULT_OCR_HEF,
    IMAGE_EXTENSIONS,
    PROJECT_ROOT,
    display_path,
)
from .web_jobs import (
    BUSY_REJECTION_MESSAGE,
    DEFAULT_HEF_BATCH_SIZE,
    DEFAULT_WORK_BLOCK_SIZE,
    TERMINAL_STATUSES,
    WebOcrJob,
    WebOcrJobManager,
)


WEB_ROOT = PROJECT_ROOT / "web"
STATIC_ROOT = WEB_ROOT / "static"
LOGO_IMAGE = STATIC_ROOT / "tinycyclops-logo.png"
THUMBNAIL_MAX_EDGE = 160
THUMBNAIL_JPEG_QUALITY = 66
CUSTOM_UPLOAD_ROOT = PROJECT_ROOT / "runs" / "web_custom_uploads"
CUSTOM_UPLOAD_MAX_IMAGES = 10
CUSTOM_UPLOAD_MAX_FILE_BYTES = 10 * 1024 * 1024
CUSTOM_UPLOAD_MAX_TOTAL_BYTES = 50 * 1024 * 1024
CUSTOM_IMAGE_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
}


class CreateJobRequest(BaseModel):
    preset: str = Field(default="icdar2015")
    limit: int | None = Field(default=None, ge=1, le=1000)
    hef_batch_size: int = Field(default=DEFAULT_HEF_BATCH_SIZE, ge=1, le=20)
    work_block_size: int = Field(default=DEFAULT_WORK_BLOCK_SIZE, ge=1, le=500)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.job_manager = WebOcrJobManager(max_queue_size=3)
    try:
        yield
    finally:
        app.state.job_manager.shutdown()


app = FastAPI(
    title="TinyCyclops OCR Workload",
    description="FastAPI service interface for the TinyCyclops HAILO OCR workload demo.",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")


def get_manager(request: Request) -> WebOcrJobManager:
    return request.app.state.job_manager


def get_job_or_404(manager: WebOcrJobManager, job_id: str) -> WebOcrJob:
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def resolve_job_image_path(job: WebOcrJob, image_name: str) -> Path:
    if Path(image_name).name != image_name:
        raise HTTPException(status_code=404, detail="Image not found")

    image_path = job.input_path / image_name
    input_path = job.input_path.resolve()
    try:
        image_path.parent.resolve().relative_to(input_path)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Image not found") from exc

    return image_path


def get_image_mime_type(image_name: str) -> str:
    suffix = Path(image_name).suffix.lower()
    return CUSTOM_IMAGE_MIME.get(suffix, "application/octet-stream")


def safe_uploaded_image_name(index: int, filename: str | None) -> str:
    raw_name = Path(filename or f"upload_{index}.jpg").name
    suffix = Path(raw_name).suffix.lower()
    stem = Path(raw_name).stem
    if suffix not in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image format: {raw_name}. Use jpg, png, or bmp.",
        )

    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("._-") or f"image-{index}"
    safe_stem = safe_stem[:64]
    return f"{index:03d}_{safe_stem}{suffix}"


def close_uploads(files: list[UploadFile]) -> None:
    for upload in files:
        upload.file.close()


@lru_cache(maxsize=1024)
def make_thumbnail(image_path: str, mtime_ns: int, file_size: int) -> bytes:
    del mtime_ns, file_size
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Image cannot be decoded")

    height, width = bgr.shape[:2]
    scale = min(THUMBNAIL_MAX_EDGE / max(width, height), 1.0)
    if scale < 1.0:
        next_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        bgr = cv2.resize(bgr, next_size, interpolation=cv2.INTER_AREA)

    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), THUMBNAIL_JPEG_QUALITY],
    )
    if not ok:
        raise ValueError("Thumbnail cannot be encoded")
    return encoded.tobytes()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    manager = get_manager(request)
    icdar = get_dataset_preset("icdar2015")
    ccpd = get_dataset_preset("ccpd")
    return {
        "status": "ok",
        "project_root": display_path(PROJECT_ROOT),
        "presets": [preset_health(preset) for preset in list_dataset_presets()],
        "icdar_images": icdar.public_preset_path,
        "icdar_images_available": icdar.preset_path.is_dir(),
        "icdar_image_count": count_images(icdar.preset_path),
        "ccpd_images": ccpd.public_preset_path,
        "ccpd_images_available": ccpd.preset_path.is_dir(),
        "ccpd_image_count": count_images(ccpd.preset_path),
        "det_hef": display_path(DEFAULT_DET_HEF),
        "det_hef_available": DEFAULT_DET_HEF.is_file(),
        "ocr_hef": display_path(DEFAULT_OCR_HEF),
        "ocr_hef_available": DEFAULT_OCR_HEF.is_file(),
        "logo_image": "assets/tinycyclops-logo.png",
        "logo_image_available": LOGO_IMAGE.is_file(),
        "jobs": manager.metrics(),
    }


@app.get("/assets/tinycyclops-logo.png")
def logo_image() -> FileResponse:
    if not LOGO_IMAGE.is_file():
        raise HTTPException(status_code=404, detail="TinyCyclops logo image not found")
    return FileResponse(LOGO_IMAGE, media_type="image/png")


@app.get("/api/metrics")
def metrics(request: Request) -> dict[str, Any]:
    return get_manager(request).metrics()


@app.get("/api/jobs")
def list_jobs(request: Request) -> dict[str, Any]:
    return {"jobs": get_manager(request).list_jobs()}


@app.post("/api/jobs", status_code=202)
def create_job(request: Request, payload: CreateJobRequest) -> dict[str, Any]:
    manager = get_manager(request)
    try:
        job = manager.create_preset_job(
            preset=payload.preset,
            limit=payload.limit,
            hef_batch_size=payload.hef_batch_size,
            work_block_size=payload.work_block_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc

    return job.snapshot()


@app.post("/api/custom-jobs", status_code=202)
def create_custom_job(
    request: Request,
    files: list[UploadFile] = File(...),
    hef_batch_size: int = Form(DEFAULT_HEF_BATCH_SIZE),
    work_block_size: int = Form(DEFAULT_WORK_BLOCK_SIZE),
) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one image.")
    if len(files) > CUSTOM_UPLOAD_MAX_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Upload up to {CUSTOM_UPLOAD_MAX_IMAGES} images at once.",
        )
    if not 1 <= hef_batch_size <= 20:
        raise HTTPException(status_code=400, detail="HEF batch must be between 1 and 20.")
    if not 1 <= work_block_size <= 500:
        raise HTTPException(status_code=400, detail="Work block must be between 1 and 500.")

    manager = get_manager(request)
    if manager.is_busy():
        raise HTTPException(
            status_code=429,
            detail=BUSY_REJECTION_MESSAGE,
        )

    stamp = uuid.uuid4().hex[:10]
    upload_dir = CUSTOM_UPLOAD_ROOT / stamp
    upload_dir.mkdir(parents=True, exist_ok=False)
    uploaded_images: list[dict[str, Any]] = []
    total_bytes = 0

    try:
        for index, upload in enumerate(files, start=1):
            image_name = safe_uploaded_image_name(index, upload.filename)
            image_bytes = upload.file.read(CUSTOM_UPLOAD_MAX_FILE_BYTES + 1)
            if len(image_bytes) > CUSTOM_UPLOAD_MAX_FILE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"{Path(upload.filename or image_name).name} exceeds 10 MiB.",
                )

            total_bytes += len(image_bytes)
            if total_bytes > CUSTOM_UPLOAD_MAX_TOTAL_BYTES:
                raise HTTPException(status_code=413, detail="Upload total exceeds 50 MiB.")

            image_path = upload_dir / image_name
            image_path.write_bytes(image_bytes)
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise HTTPException(
                    status_code=415,
                    detail=f"{Path(upload.filename or image_name).name} cannot be decoded as an image.",
                )

            height, width = bgr.shape[:2]
            uploaded_images.append(
                {
                    "image_name": image_name,
                    "source_name": Path(upload.filename or image_name).name,
                    "width": width,
                    "height": height,
                    "size_bytes": len(image_bytes),
                }
            )

        try:
            job = manager.create_custom_job(
                input_path=upload_dir,
                uploaded_images=uploaded_images,
                hef_batch_size=hef_batch_size,
                work_block_size=work_block_size,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
    except Exception:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise
    finally:
        close_uploads(files)

    return job.snapshot()


@app.get("/api/jobs/{job_id}")
def get_job(request: Request, job_id: str) -> dict[str, Any]:
    return get_job_or_404(get_manager(request), job_id).snapshot()


@app.get("/api/jobs/{job_id}/events")
def job_events(request: Request, job_id: str, after: int = 0) -> StreamingResponse:
    job = get_job_or_404(get_manager(request), job_id)

    def event_stream():
        cursor = after
        yield "retry: 1000\n\n"

        while True:
            events = job.wait_for_events(cursor, timeout=15)
            if not events:
                yield ": keepalive\n\n"
                continue

            for event in events:
                cursor = max(cursor, int(event["seq"]))
                payload = json.dumps(event, ensure_ascii=False)
                yield f"id: {event['seq']}\ndata: {payload}\n\n"

            snapshot = job.snapshot()
            if snapshot["status"] in TERMINAL_STATUSES and cursor >= job.last_event_seq():
                return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/jobs/{job_id}/results")
def job_results(request: Request, job_id: str) -> dict[str, Any]:
    job = get_job_or_404(get_manager(request), job_id)
    snapshot = job.snapshot()
    if snapshot["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Job is not completed: {snapshot['status']}")

    return {
        "job": snapshot,
        "summary": job.summary,
        "results": job.results or [],
    }


@app.get("/api/jobs/{job_id}/images/{image_name}")
def job_image(request: Request, job_id: str, image_name: str) -> FileResponse:
    job = get_job_or_404(get_manager(request), job_id)
    image_path = resolve_job_image_path(job, image_name)
    if image_path.is_file():
        return FileResponse(image_path, filename=image_path.name)

    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/api/jobs/{job_id}/thumbnails/{image_name}")
def job_thumbnail(request: Request, job_id: str, image_name: str) -> Response:
    job = get_job_or_404(get_manager(request), job_id)
    image_path = resolve_job_image_path(job, image_name)
    if image_path.is_file():
        stat = image_path.stat()
        try:
            thumbnail = make_thumbnail(str(image_path), stat.st_mtime_ns, stat.st_size)
        except ValueError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
    else:
        raise HTTPException(status_code=404, detail="Image not found")

    return Response(
        content=thumbnail,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=300"},
    )


@app.get("/api/jobs/{job_id}/artifacts/{artifact_name}")
def job_artifact(request: Request, job_id: str, artifact_name: str) -> FileResponse:
    job = get_job_or_404(get_manager(request), job_id)
    artifact_path = job.artifact_path(artifact_name)
    if artifact_path is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    return FileResponse(artifact_path, filename=artifact_path.name)
