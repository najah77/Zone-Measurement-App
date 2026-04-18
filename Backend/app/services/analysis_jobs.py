from __future__ import annotations

import logging
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Dict

from fastapi import HTTPException

from app.core.config import settings
from app.models.schemas import AnalysisSubmissionResponse
from app.services.analysis_store import (
    ensure_storage_root,
    load_analysis_job_status,
    initialize_analysis_job,
    load_uploaded_image_bytes,
    save_analysis_failure,
    save_analysis_record,
    update_analysis_job_status,
)
from app.services.ast_processor import run_ast_analysis

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=max(1, getattr(settings, "ANALYSIS_WORKERS", 2)))
_active_jobs: Dict[str, Future[None]] = {}
_active_jobs_lock = Lock()


def _remove_active_job(analysis_id: str) -> None:
    with _active_jobs_lock:
        _active_jobs.pop(analysis_id, None)


def _progress_reporter(analysis_id: str, image_filename: str | None):
    def _report(progress: float, current_stage: str, message: str, timings):
        update_analysis_job_status(
            analysis_id,
            status="processing",
            progress=progress,
            current_stage=current_stage,
            message=message,
            timings=timings or {},
            image_filename=image_filename,
            result_available=False,
        )

    return _report


def _run_analysis_job(analysis_id: str, image_filename: str | None, include_debug_artifacts: bool) -> None:
    stage_timings: Dict[str, float] = {}
    try:
        update_analysis_job_status(
            analysis_id,
            status="processing",
            progress=0.05,
            current_stage="processing",
            message="Analysis job started.",
            timings=stage_timings,
            image_filename=image_filename,
            result_available=False,
        )
        image_bytes = load_uploaded_image_bytes(analysis_id)
        analysis = run_ast_analysis(
            image_bytes=image_bytes,
            analysis_id=analysis_id,
            image_filename=image_filename,
            include_debug_artifacts=include_debug_artifacts,
            progress_callback=_progress_reporter(analysis_id, image_filename),
            stage_timings=stage_timings,
        )
        save_analysis_record(
            analysis,
            image_bytes=image_bytes,
            image_filename=image_filename,
            stage_timings=stage_timings,
        )
    except HTTPException as exc:
        message = str(exc.detail)
        logger.warning("Analysis job %s failed with HTTP error: %s", analysis_id, message)
        save_analysis_failure(
            analysis_id,
            message=message,
            current_stage="failed",
            timings=stage_timings,
            image_filename=image_filename,
        )
    except Exception as exc:  # pragma: no cover - defensive safeguard
        logger.exception("Analysis job %s crashed.", analysis_id)
        message = f"{exc.__class__.__name__}: {exc}"
        stage_timings["traceback_captured"] = 1.0
        save_analysis_failure(
            analysis_id,
            message=message,
            current_stage="failed",
            timings=stage_timings,
            image_filename=image_filename,
            technical_details=traceback.format_exc(limit=10),
        )
    finally:
        _remove_active_job(analysis_id)


def submit_analysis_job(
    image_bytes: bytes,
    *,
    image_filename: str | None = None,
    include_debug_artifacts: bool = False,
) -> AnalysisSubmissionResponse:
    analysis_id = str(uuid.uuid4())
    submission = initialize_analysis_job(
        analysis_id,
        image_bytes=image_bytes,
        image_filename=image_filename,
        include_debug_artifacts=include_debug_artifacts,
    )
    future = _executor.submit(_run_analysis_job, analysis_id, image_filename, include_debug_artifacts)
    with _active_jobs_lock:
        _active_jobs[analysis_id] = future
    return submission


def recover_interrupted_analysis_jobs() -> int:
    recovered = 0
    storage_root = ensure_storage_root()
    for analysis_dir in storage_root.iterdir():
        if not analysis_dir.is_dir():
            continue
        analysis_id = analysis_dir.name
        try:
            status = load_analysis_job_status(analysis_id)
        except HTTPException:
            continue
        if status.status not in {"queued", "processing"}:
            continue
        save_analysis_failure(
            analysis_id,
            message="Analysis was interrupted before completion. Please retry the upload.",
            current_stage="interrupted",
            timings=status.timings,
            image_filename=status.image_filename,
            technical_details="The backend restarted before the background analysis job completed.",
        )
        recovered += 1

    if recovered:
        logger.warning("Marked %s interrupted analysis job(s) as failed during startup recovery.", recovered)
    return recovered
