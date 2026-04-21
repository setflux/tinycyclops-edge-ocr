from __future__ import annotations

import contextlib
import json
import queue
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset_registry import (
    get_dataset_preset,
    has_preset_images,
    materialize_symlink_workload,
)
from .runtime import (
    DEFAULT_DET_HEF,
    DEFAULT_OCR_HEF,
    PROJECT_ROOT,
    OcrRunArtifacts,
    artifact_paths,
    display_path,
    read_results_jsonl,
    read_summary_json,
)


TERMINAL_STATUSES = {"completed", "failed"}
DEFAULT_HEF_BATCH_SIZE = 10
DEFAULT_WORK_BLOCK_SIZE = 20
BUSY_REJECTION_MESSAGE = "Tiny is already helping someone else. Please try again in a moment."


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def public_path_value(value: Any) -> Any:
    if value is None or value == "":
        return value
    return display_path(Path(str(value)))


def public_ocr_result(result: dict[str, Any]) -> dict[str, Any]:
    public_result = dict(result)
    if "image" in public_result:
        public_result["image"] = public_path_value(public_result["image"])
    return public_result


def public_summary(summary: dict[str, Any]) -> dict[str, Any]:
    public_summary_value = dict(summary)
    for key in ("input_path", "output_dir", "det_hef", "ocr_hef"):
        if key in public_summary_value:
            public_summary_value[key] = public_path_value(public_summary_value[key])

    artifacts = public_summary_value.get("artifacts")
    if isinstance(artifacts, dict):
        public_summary_value["artifacts"] = {
            key: public_path_value(value) for key, value in artifacts.items()
        }
    return public_summary_value


@dataclass
class WebOcrJob:
    job_id: str
    input_path: Path
    output_dir: Path
    limit: int | None
    hef_batch_size: int
    work_block_size: int
    preset: str = "icdar2015"
    status: str = "queued"
    created_at: str = field(default_factory=utc_now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    image_count: int = 0
    processed_image_count: int = 0
    detection_count: int = 0
    nonempty_image_count: int = 0
    fps: float = 0.0
    elapsed_seconds: float = 0.0
    error: str | None = None
    latest_image: dict[str, Any] | None = None
    uploaded_images: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] | None = None
    results: list[dict[str, Any]] | None = None
    artifacts: OcrRunArtifacts | None = None
    randomized_workload: dict[str, Any] | None = None
    randomize_source_path: Path | None = None
    randomize_index_path: Path | None = None
    randomize_count: int | None = None
    randomize_strategy: str = "balanced"
    randomize_seed: str | None = None
    run_stdout_log: Path | None = None
    run_progress_log: Path | None = None
    child_stdout_log: Path | None = None
    child_stderr_log: Path | None = None
    _events: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=2000), repr=False)
    _condition: threading.Condition = field(default_factory=threading.Condition, repr=False)

    def append_event(self, event_type: str, **payload: Any) -> dict[str, Any]:
        with self._condition:
            event = {
                "seq": self._events[-1]["seq"] + 1 if self._events else 1,
                "type": event_type,
                "job_id": self.job_id,
                "timestamp": utc_now_iso(),
                **payload,
            }
            self._events.append(event)
            self._condition.notify_all()
            return event

    def update_from_pipeline_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("event", "progress"))
        payload = {key: value for key, value in event.items() if key != "event"}
        if "image" in payload:
            payload["image"] = public_path_value(payload["image"])

        with self._condition:
            if event_type in {"discovered", "images_loaded", "started"}:
                self.image_count = int(payload.get("image_count") or self.image_count or 0)
            elif event_type == "image_completed":
                self.image_count = int(payload.get("image_count") or self.image_count or 0)
                self.processed_image_count = int(
                    payload.get("processed_image_count") or self.processed_image_count
                )
                self.detection_count = int(payload.get("total_detection_count") or self.detection_count)
                self.nonempty_image_count = int(payload.get("nonempty_image_count") or self.nonempty_image_count)
                self.fps = float(payload.get("fps") or self.fps or 0.0)
                self.elapsed_seconds = float(payload.get("elapsed_seconds") or self.elapsed_seconds or 0.0)
                self.latest_image = {
                    "image": payload.get("image"),
                    "image_name": payload.get("image_name"),
                    "width": payload.get("width"),
                    "height": payload.get("height"),
                    "detection_count": payload.get("detection_count"),
                    "text_preview": payload.get("text_preview"),
                }

        self.append_event(event_type, **payload)

    def mark_running(self) -> None:
        with self._condition:
            self.status = "running"
            self.started_at = utc_now_iso()
        self.append_event("job_started", status=self.status)

    def mark_randomizing(self) -> None:
        self.append_event(
            "randomizing_workload",
            status=self.status,
            message="Randomizing Workload",
            image_count=self.randomize_count or self.image_count or 0,
            processed_image_count=0,
        )

    def mark_workload_randomized(self, manifest: dict[str, Any]) -> None:
        public_manifest = {key: value for key, value in manifest.items() if key != "images"}
        with self._condition:
            self.randomized_workload = public_manifest
            self.image_count = int(manifest.get("image_count") or self.image_count or 0)
        self.append_event(
            "workload_randomized",
            status=self.status,
            message="Processing OCR",
            image_count=self.image_count,
            processed_image_count=0,
            randomized_workload=public_manifest,
        )

    def set_process_logs(
        self,
        *,
        run_stdout_log: Path,
        run_progress_log: Path,
        child_stdout_log: Path,
        child_stderr_log: Path,
    ) -> None:
        with self._condition:
            self.run_stdout_log = run_stdout_log
            self.run_progress_log = run_progress_log
            self.child_stdout_log = child_stdout_log
            self.child_stderr_log = child_stderr_log

    def mark_completed(
        self,
        results: list[dict[str, Any]],
        summary: dict[str, Any],
        artifacts: OcrRunArtifacts,
        run_stdout_log: Path,
    ) -> None:
        with self._condition:
            self.status = "completed"
            self.finished_at = utc_now_iso()
            self.results = [public_ocr_result(result) for result in results]
            self.summary = public_summary(summary)
            self.artifacts = artifacts
            self.run_stdout_log = run_stdout_log
            self.image_count = int(summary.get("image_count") or self.image_count or 0)
            self.processed_image_count = self.image_count
            self.detection_count = int(summary.get("detection_count") or self.detection_count or 0)
            self.nonempty_image_count = int(summary.get("nonempty_image_count") or self.nonempty_image_count or 0)
            self.fps = float(summary.get("fps") or self.fps or 0.0)
            self.elapsed_seconds = float(summary.get("elapsed_seconds") or self.elapsed_seconds or 0.0)
        self.append_event(
            "job_completed",
            status=self.status,
            summary=self.summary or {},
            artifacts=self.artifact_map(),
        )

    def mark_failed(self, error: str) -> None:
        with self._condition:
            self.status = "failed"
            self.finished_at = utc_now_iso()
            self.error = error
        self.append_event("job_failed", status=self.status, error=error)

    def _artifact_paths(self) -> dict[str, Path]:
        if self.artifacts is None:
            return {}

        artifacts = {
            "summary.json": self.artifacts.summary_json,
            "results.jsonl": self.artifacts.results_jsonl,
            "detections.csv": self.artifacts.detections_csv,
            "full_text.txt": self.artifacts.full_text_txt,
        }
        if self.run_stdout_log is not None:
            artifacts["run_stdout.log"] = self.run_stdout_log
        if self.run_progress_log is not None:
            artifacts["progress.jsonl"] = self.run_progress_log
        if self.child_stdout_log is not None:
            artifacts["child_stdout.log"] = self.child_stdout_log
        if self.child_stderr_log is not None:
            artifacts["child_stderr.log"] = self.child_stderr_log
        workload_manifest = self.output_dir / "workload_manifest.json"
        if workload_manifest.is_file():
            artifacts["workload_manifest.json"] = workload_manifest

        return artifacts

    def artifact_map(self) -> dict[str, str]:
        return {name: display_path(path) for name, path in self._artifact_paths().items()}

    def artifact_path(self, artifact_name: str) -> Path | None:
        if Path(artifact_name).name != artifact_name:
            return None

        artifact_map = self._artifact_paths()
        artifact_path = artifact_map.get(artifact_name)
        if artifact_path is None:
            return None

        return artifact_path if artifact_path.is_file() else None

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "job_id": self.job_id,
                "preset": self.preset,
                "status": self.status,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "input_path": display_path(self.input_path),
                "output_dir": display_path(self.output_dir),
                "limit": self.limit,
                "hef_batch_size": self.hef_batch_size,
                "work_block_size": self.work_block_size,
                "image_count": self.image_count,
                "processed_image_count": self.processed_image_count,
                "detection_count": self.detection_count,
                "nonempty_image_count": self.nonempty_image_count,
                "fps": self.fps,
                "elapsed_seconds": self.elapsed_seconds,
                "error": self.error,
                "latest_image": self.latest_image,
                "uploaded_images": self.uploaded_images,
                "randomized_workload": self.randomized_workload,
                "event_seq": self._events[-1]["seq"] if self._events else 0,
                "artifacts": self.artifact_map(),
            }

    def wait_for_events(self, after_seq: int, timeout: float = 15.0) -> list[dict[str, Any]]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                events = [event for event in self._events if event["seq"] > after_seq]
                if events or self.status in TERMINAL_STATUSES:
                    return events

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []

                self._condition.wait(timeout=remaining)

    def last_event_seq(self) -> int:
        with self._condition:
            return self._events[-1]["seq"] if self._events else 0


class WebOcrJobManager:
    def __init__(self, max_queue_size: int = 3, reject_when_busy: bool = True) -> None:
        self._jobs: dict[str, WebOcrJob] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[WebOcrJob | None] = queue.Queue(maxsize=max_queue_size)
        self._reject_when_busy = reject_when_busy
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="tinycyclops-web-job-worker",
            daemon=True,
        )
        self._worker_thread.start()

    def create_preset_job(
        self,
        *,
        preset: str = "icdar2015",
        limit: int | None = None,
        hef_batch_size: int = DEFAULT_HEF_BATCH_SIZE,
        work_block_size: int = DEFAULT_WORK_BLOCK_SIZE,
    ) -> WebOcrJob:
        try:
            preset_config = get_dataset_preset(preset)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc

        job_id = uuid.uuid4().hex[:12]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "runs" / f"{preset_config.run_prefix}_{stamp}_{job_id}"
        input_path = preset_config.preset_path
        job_limit = limit
        randomize_source_path: Path | None = None
        randomize_index_path: Path | None = None
        randomize_count: int | None = None
        randomize_seed: str | None = None

        if preset_config.supports_randomize:
            if preset_config.source_path is None or preset_config.index_path is None:
                raise ValueError(f"{preset_config.label} is not configured for runtime randomization.")
            if not preset_config.source_path.is_dir():
                raise ValueError(f"{preset_config.label} source is not prepared: {preset_config.public_source_path}")
            if not preset_config.index_path.is_file():
                raise ValueError(f"{preset_config.label} index is not prepared: {preset_config.public_index_path}")

            randomize_count = limit or preset_config.default_sample_size
            if randomize_count is None:
                raise ValueError(f"{preset_config.label} default random sample size is not configured.")
            randomize_source_path = preset_config.source_path
            randomize_index_path = preset_config.index_path
            randomize_seed = secrets.token_hex(8)
            input_path = output_dir / "input_images"
            job_limit = None
        elif not has_preset_images(preset_config):
            raise ValueError(
                f"{preset_config.label} dataset is not prepared yet. "
                f"Prepare {preset_config.public_preset_path} before running this preset."
            )

        job = WebOcrJob(
            job_id=job_id,
            input_path=input_path,
            output_dir=output_dir,
            limit=job_limit,
            hef_batch_size=hef_batch_size,
            work_block_size=work_block_size,
            preset=preset,
            image_count=randomize_count or 0,
            randomize_source_path=randomize_source_path,
            randomize_index_path=randomize_index_path,
            randomize_count=randomize_count,
            randomize_seed=randomize_seed,
        )
        job.append_event("job_queued", status=job.status)

        self._register_and_enqueue(job)
        return job

    def create_custom_job(
        self,
        *,
        input_path: Path,
        uploaded_images: list[dict[str, Any]],
        hef_batch_size: int = DEFAULT_HEF_BATCH_SIZE,
        work_block_size: int = DEFAULT_WORK_BLOCK_SIZE,
    ) -> WebOcrJob:
        job_id = uuid.uuid4().hex[:12]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job = WebOcrJob(
            job_id=job_id,
            input_path=input_path,
            output_dir=PROJECT_ROOT / "runs" / f"web_custom_{stamp}_{job_id}",
            limit=None,
            hef_batch_size=hef_batch_size,
            work_block_size=work_block_size,
            preset="custom",
            image_count=len(uploaded_images),
            uploaded_images=uploaded_images,
        )
        job.append_event("job_queued", status=job.status, image_count=job.image_count)

        self._register_and_enqueue(job)
        return job

    def is_busy(self) -> bool:
        with self._lock:
            return self._has_active_job_locked()

    def _register_and_enqueue(self, job: WebOcrJob) -> None:
        with self._lock:
            if self._reject_when_busy and self._has_active_job_locked():
                raise RuntimeError(BUSY_REJECTION_MESSAGE)
            self._jobs[job.job_id] = job

        try:
            self._queue.put_nowait(job)
        except queue.Full as exc:
            with self._lock:
                self._jobs.pop(job.job_id, None)
            raise RuntimeError("TinyCyclops job queue is full") from exc

    def _has_active_job_locked(self) -> bool:
        return any(job.status not in TERMINAL_STATUSES for job in self._jobs.values())

    def get_job(self, job_id: str) -> WebOcrJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        return [job.snapshot() for job in sorted(jobs, key=lambda item: item.created_at, reverse=True)]

    def metrics(self) -> dict[str, Any]:
        jobs = self.list_jobs()
        return {
            "job_count": len(jobs),
            "queued_job_count": sum(1 for job in jobs if job["status"] == "queued"),
            "running_job_count": sum(1 for job in jobs if job["status"] == "running"),
            "completed_job_count": sum(1 for job in jobs if job["status"] == "completed"),
            "failed_job_count": sum(1 for job in jobs if job["status"] == "failed"),
            "queue_capacity": self._queue.maxsize,
            "queue_size": self._queue.qsize(),
            "latest_job": jobs[0] if jobs else None,
        }

    def shutdown(self) -> None:
        self._stop_event.set()
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)
        self._worker_thread.join(timeout=5)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            job = self._queue.get()
            try:
                if job is None:
                    return
                self._run_job(job)
            finally:
                self._queue.task_done()

    def _run_job(self, job: WebOcrJob) -> None:
        job.mark_running()
        job.output_dir.mkdir(parents=True, exist_ok=True)
        run_stdout_log = job.output_dir / "run_stdout.log"
        progress_jsonl = job.output_dir / "progress.jsonl"
        child_stdout_log = job.output_dir / "child_stdout.log"
        child_stderr_log = job.output_dir / "child_stderr.log"
        job_spec_json = job.output_dir / "job_spec.json"
        job.set_process_logs(
            run_stdout_log=run_stdout_log,
            run_progress_log=progress_jsonl,
            child_stdout_log=child_stdout_log,
            child_stderr_log=child_stderr_log,
        )

        try:
            if job.randomize_source_path is not None and job.randomize_index_path is not None:
                manifest = self._materialize_randomized_workload(job)
                (job.output_dir / "workload_manifest.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

            job_spec = {
                "job_id": job.job_id,
                "preset": job.preset,
                "input_path": str(job.input_path),
                "output_dir": str(job.output_dir),
                "det_hef": str(DEFAULT_DET_HEF),
                "ocr_hef": str(DEFAULT_OCR_HEF),
                "limit": job.limit,
                "hef_batch_size": job.hef_batch_size,
                "work_block_size": job.work_block_size,
                "use_corrector": False,
                "run_stdout_log": str(run_stdout_log),
                "progress_jsonl": str(progress_jsonl),
            }
            job_spec_json.write_text(json.dumps(job_spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            progress_jsonl.touch(exist_ok=True)

            return_code = self._run_child_process(
                job=job,
                job_spec_json=job_spec_json,
                progress_jsonl=progress_jsonl,
                child_stdout_log=child_stdout_log,
                child_stderr_log=child_stderr_log,
            )
            if return_code != 0:
                error_tail = self._read_log_tail(child_stderr_log)
                raise RuntimeError(
                    f"TinyCyclops OCR child process failed with exit code {return_code}."
                    + (f"\n{error_tail}" if error_tail else "")
                )

            artifacts = artifact_paths(job.output_dir)
            summary = read_summary_json(artifacts.summary_json)
            results = read_results_jsonl(artifacts.results_jsonl)
            job.mark_completed(results, summary, artifacts, run_stdout_log)
        except Exception as exc:
            job.mark_failed(str(exc))
        finally:
            if job.preset == "custom":
                shutil.rmtree(job.input_path, ignore_errors=True)

    def _materialize_randomized_workload(self, job: WebOcrJob) -> dict[str, object]:
        if job.randomize_source_path is None or job.randomize_index_path is None:
            raise RuntimeError("Randomized workload is not configured.")
        if job.randomize_count is None or job.randomize_count < 1:
            raise RuntimeError("Randomized workload image count is invalid.")

        job.mark_randomizing()
        manifest = materialize_symlink_workload(
            source_path=job.randomize_source_path,
            index_path=job.randomize_index_path,
            target_path=job.input_path,
            limit=job.randomize_count,
            strategy=job.randomize_strategy,
            seed=job.randomize_seed or secrets.token_hex(8),
        )
        job.mark_workload_randomized(manifest)
        return manifest

    def _run_child_process(
        self,
        *,
        job: WebOcrJob,
        job_spec_json: Path,
        progress_jsonl: Path,
        child_stdout_log: Path,
        child_stderr_log: Path,
    ) -> int:
        command = [
            sys.executable,
            "-m",
            "tinycyclops_ocr.ocr_child",
            "--job-spec",
            str(job_spec_json),
        ]

        with (
            child_stdout_log.open("w", encoding="utf-8") as stdout_fp,
            child_stderr_log.open("w", encoding="utf-8") as stderr_fp,
            progress_jsonl.open("r", encoding="utf-8") as progress_fp,
        ):
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=stdout_fp,
                stderr=stderr_fp,
                text=True,
            )
            try:
                while process.poll() is None:
                    self._drain_progress_events(job, progress_fp)
                    if self._stop_event.wait(0.1):
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
                        raise RuntimeError("TinyCyclops OCR child process was stopped.")
                self._drain_progress_events(job, progress_fp)
                return int(process.returncode or 0)
            finally:
                self._drain_progress_events(job, progress_fp)

    def _drain_progress_events(self, job: WebOcrJob, progress_fp) -> None:
        while True:
            line = progress_fp.readline()
            if not line:
                return
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                job.append_event("progress_parse_warning", error=str(exc), raw=line[:500])
                continue
            job.update_from_pipeline_event(event)

    def _read_log_tail(self, path: Path, max_bytes: int = 4000) -> str:
        if not path.is_file():
            return ""
        size = path.stat().st_size
        with path.open("rb") as fp:
            if size > max_bytes:
                fp.seek(size - max_bytes)
            return fp.read().decode("utf-8", errors="replace").strip()
