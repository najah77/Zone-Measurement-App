from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, Optional

import cv2
import numpy as np
from fastapi import HTTPException

from app.core.config import settings
from app.models.schemas import (
    AnalysisErrorRecord,
    AnalysisJobStatusResponse,
    AnalysisResponse,
    AnalysisSubmissionResponse,
    PersistedAnalysisRecord,
    SaveReviewRequest,
)
from app.utils.measurement import normalize_corrected_diameter


def _model_dump(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def ensure_storage_root() -> Path:
    settings.STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    return settings.STORAGE_ROOT


def _analysis_dir(analysis_id: str) -> Path:
    return ensure_storage_root() / analysis_id


def _record_path(analysis_id: str) -> Path:
    return _analysis_dir(analysis_id) / "analysis.json"


def _result_path(analysis_id: str) -> Path:
    return _analysis_dir(analysis_id) / "result.json"


def _status_path(analysis_id: str) -> Path:
    return _analysis_dir(analysis_id) / "status.json"


def _error_path(analysis_id: str) -> Path:
    return _analysis_dir(analysis_id) / "error.json"


def _request_path(analysis_id: str) -> Path:
    return _analysis_dir(analysis_id) / "request.json"


def _image_path(analysis_id: str) -> Path:
    return _analysis_dir(analysis_id) / "original_upload.bin"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temp_path.replace(path)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _delete_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _status_url(analysis_id: str) -> str:
    return f"{settings.API_PREFIX}/analyze/{analysis_id}/status"


def _result_url(analysis_id: str) -> str:
    return f"{settings.API_PREFIX}/analyze/{analysis_id}/result"


def _build_status_payload(
    analysis_id: str,
    *,
    status: str,
    created_at: datetime,
    updated_at: datetime,
    image_filename: str | None,
    message: str,
    progress: float,
    current_stage: str | None,
    error: str | None,
    result_available: bool,
    timings: Dict[str, float] | None = None,
) -> AnalysisJobStatusResponse:
    return AnalysisJobStatusResponse(
        analysis_id=analysis_id,
        status=status,
        created_at=created_at,
        updated_at=updated_at,
        image_filename=image_filename,
        message=message,
        progress=max(0.0, min(1.0, progress)),
        current_stage=current_stage,
        error=error,
        result_available=result_available,
        timings={key: round(float(value), 3) for key, value in (timings or {}).items()},
        status_url=_status_url(analysis_id),
        result_url=_result_url(analysis_id),
    )


def _encode_image_to_base64(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _persist_debug_artifacts(directory: Path, analysis: AnalysisResponse) -> None:
    if not analysis.debug_artifacts:
        return

    for key, value in list(analysis.debug_artifacts.items()):
        if not key.endswith("_base64") or not isinstance(value, str) or not value:
            continue
        try:
            image_bytes = base64.b64decode(value)
        except (ValueError, TypeError):
            continue

        suffix = key[: -len("_base64")]
        target = directory / f"{suffix}.png"
        target.write_bytes(image_bytes)
        analysis.debug_artifacts[f"{suffix}_path"] = str(target)


def initialize_analysis_job(
    analysis_id: str,
    image_bytes: bytes,
    image_filename: str | None = None,
    include_debug_artifacts: bool = False,
) -> AnalysisSubmissionResponse:
    directory = _analysis_dir(analysis_id)
    directory.mkdir(parents=True, exist_ok=True)

    _image_path(analysis_id).write_bytes(image_bytes)
    request_payload = {
        "analysis_id": analysis_id,
        "image_filename": image_filename,
        "include_debug_artifacts": include_debug_artifacts,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(_request_path(analysis_id), request_payload)

    now = datetime.now(timezone.utc)
    status = _build_status_payload(
        analysis_id,
        status="queued",
        created_at=now,
        updated_at=now,
        image_filename=image_filename,
        message="Upload received. Analysis job queued.",
        progress=0.0,
        current_stage="queued",
        error=None,
        result_available=False,
    )
    _write_json(_status_path(analysis_id), _model_dump(status))
    _delete_if_exists(_error_path(analysis_id))
    _delete_if_exists(_result_path(analysis_id))

    return AnalysisSubmissionResponse(
        analysis_id=analysis_id,
        status=status.status,
        created_at=status.created_at,
        message=status.message,
        status_url=_status_url(analysis_id),
        result_url=_result_url(analysis_id),
    )


def update_analysis_job_status(
    analysis_id: str,
    *,
    status: str | None = None,
    message: str | None = None,
    progress: float | None = None,
    current_stage: str | None = None,
    error: str | None = None,
    result_available: bool | None = None,
    timings: Dict[str, float] | None = None,
    image_filename: str | None = None,
) -> AnalysisJobStatusResponse:
    existing = load_analysis_job_status(analysis_id)
    updated = _build_status_payload(
        analysis_id,
        status=status or existing.status,
        created_at=existing.created_at,
        updated_at=datetime.now(timezone.utc),
        image_filename=image_filename or existing.image_filename,
        message=message if message is not None else existing.message,
        progress=progress if progress is not None else existing.progress,
        current_stage=current_stage if current_stage is not None else existing.current_stage,
        error=error if error is not None else existing.error,
        result_available=result_available if result_available is not None else existing.result_available,
        timings={**existing.timings, **(timings or {})},
    )
    _write_json(_status_path(analysis_id), _model_dump(updated))
    return updated


def save_analysis_failure(
    analysis_id: str,
    *,
    message: str,
    current_stage: str | None = None,
    timings: Dict[str, float] | None = None,
    image_filename: str | None = None,
    technical_details: str | None = None,
) -> AnalysisJobStatusResponse:
    failure = AnalysisErrorRecord(
        analysis_id=analysis_id,
        failed_at=datetime.now(timezone.utc),
        image_filename=image_filename,
        current_stage=current_stage,
        message=message,
        technical_details=technical_details,
        timings={key: round(float(value), 3) for key, value in (timings or {}).items()},
    )
    _write_json(_error_path(analysis_id), _model_dump(failure))
    _delete_if_exists(_result_path(analysis_id))
    return update_analysis_job_status(
        analysis_id,
        status="failed",
        message=message,
        progress=1.0,
        current_stage=current_stage or "failed",
        error=message,
        result_available=False,
        timings=timings or {},
        image_filename=image_filename,
    )


def load_analysis_job_status(analysis_id: str) -> AnalysisJobStatusResponse:
    status_path = _status_path(analysis_id)
    if status_path.exists():
        return AnalysisJobStatusResponse(**_load_json(status_path))

    result_path = _result_path(analysis_id)
    if result_path.exists():
        analysis = AnalysisResponse(**_load_json(result_path))
        return _build_status_payload(
            analysis_id,
            status="completed",
            created_at=analysis.created_at,
            updated_at=datetime.now(timezone.utc),
            image_filename=analysis.image_filename,
            message="Analysis completed.",
            progress=1.0,
            current_stage="completed",
            error=None,
            result_available=True,
        )

    record_path = _record_path(analysis_id)
    if record_path.exists():
        record = PersistedAnalysisRecord(**_load_json(record_path))
        return _build_status_payload(
            analysis_id,
            status="completed",
            created_at=record.analysis.created_at,
            updated_at=record.updated_at,
            image_filename=record.analysis.image_filename,
            message="Analysis completed.",
            progress=1.0,
            current_stage="completed",
            error=None,
            result_available=True,
        )

    error_path = _error_path(analysis_id)
    if error_path.exists():
        failure = AnalysisErrorRecord(**_load_json(error_path))
        return _build_status_payload(
            analysis_id,
            status="failed",
            created_at=failure.failed_at,
            updated_at=failure.failed_at,
            image_filename=failure.image_filename,
            message=failure.message,
            progress=1.0,
            current_stage=failure.current_stage,
            error=failure.message,
            result_available=False,
            timings=failure.timings,
        )

    raise HTTPException(status_code=404, detail="Analysis job not found")


def load_analysis_result(analysis_id: str) -> AnalysisResponse:
    result_path = _result_path(analysis_id)
    if result_path.exists():
        return AnalysisResponse(**_load_json(result_path))

    record_path = _record_path(analysis_id)
    if record_path.exists():
        return PersistedAnalysisRecord(**_load_json(record_path)).analysis

    status = load_analysis_job_status(analysis_id)
    if status.status in {"queued", "processing"}:
        raise HTTPException(
            status_code=409,
            detail={
                "status": status.status,
                "message": status.message or "Analysis is still in progress.",
            },
        )
    if status.status == "failed":
        raise HTTPException(
            status_code=409,
            detail={
                "status": status.status,
                "message": status.error or status.message or "Analysis failed.",
            },
        )
    raise HTTPException(status_code=404, detail="Analysis result not found")


def save_analysis_record(
    analysis: AnalysisResponse,
    image_bytes: bytes,
    image_filename: str | None = None,
    stage_timings: Dict[str, float] | None = None,
) -> AnalysisResponse:
    storage_started = perf_counter()
    directory = _analysis_dir(analysis.analysis_id)
    directory.mkdir(parents=True, exist_ok=True)

    image_path = _image_path(analysis.analysis_id)
    if not image_path.exists():
        image_path.write_bytes(image_bytes)
    _persist_debug_artifacts(directory, analysis)

    record = PersistedAnalysisRecord(
        analysis=analysis,
        storage={
            "image_path": str(image_path),
            "image_filename": image_filename or analysis.image_filename,
        },
        updated_at=datetime.now(timezone.utc),
    )
    _write_json(_record_path(analysis.analysis_id), _model_dump(record))
    _write_json(_result_path(analysis.analysis_id), _model_dump(analysis))
    _delete_if_exists(_error_path(analysis.analysis_id))

    if stage_timings is not None:
        stage_timings["storage_write_seconds"] = perf_counter() - storage_started
        if "total_runtime_seconds" in stage_timings:
            stage_timings["total_runtime_with_persistence_seconds"] = (
                stage_timings["total_runtime_seconds"] + stage_timings["storage_write_seconds"]
            )

    completed_status = _build_status_payload(
        analysis.analysis_id,
        status="completed",
        created_at=analysis.created_at,
        updated_at=datetime.now(timezone.utc),
        image_filename=image_filename or analysis.image_filename,
        message="Analysis completed.",
        progress=1.0,
        current_stage="completed",
        error=None,
        result_available=True,
        timings=stage_timings or {},
    )
    _write_json(_status_path(analysis.analysis_id), _model_dump(completed_status))
    return analysis


def load_analysis_record(analysis_id: str) -> PersistedAnalysisRecord:
    path = _record_path(analysis_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Analysis record not found")
    data = _load_json(path)
    return PersistedAnalysisRecord(**data)


def load_uploaded_image_bytes(analysis_id: str) -> bytes:
    path = _image_path(analysis_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Uploaded image not found for this analysis job")
    return path.read_bytes()


def load_request_metadata(analysis_id: str) -> Dict[str, Any]:
    path = _request_path(analysis_id)
    if not path.exists():
        return {}
    return _load_json(path)


def save_debug_image(analysis_id: str, file_name: str, image: np.ndarray) -> str:
    directory = _analysis_dir(analysis_id)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / file_name
    cv2.imwrite(str(target), image)
    return str(target)


def apply_review_update(analysis_id: str, payload: SaveReviewRequest) -> AnalysisResponse:
    record = load_analysis_record(analysis_id)
    discs_by_id = {disc.disc_id: disc for disc in record.analysis.results}

    for update in payload.discs:
        disc = discs_by_id.get(update.disc_id)
        if disc is None:
            raise HTTPException(status_code=400, detail=f"Unknown disc id: {update.disc_id}")

        if update.corrected_diameter_mm is not None:
            final_diameter = normalize_corrected_diameter(update.corrected_diameter_mm)
            disc.corrected_diameter_mm = round(final_diameter, 2)
            disc.final_diameter_mm = round(final_diameter, 2)
            disc.source = "manual"
            disc.status = "corrected"
            disc.review_required = False

        if update.corrected_code:
            disc.final_code = update.corrected_code.strip().upper()
            disc.source = "manual"
            disc.status = "corrected"
            disc.review_required = False
            disc.label_decision_source = "manual_confirmation"
            disc.label_selection_reason = "Operator manually confirmed or corrected the antibiotic code."
            if disc.final_code and disc.final_code not in disc.label_candidates:
                disc.label_candidates = [disc.final_code, *disc.label_candidates][:8]

        if update.operator_note:
            disc.operator_note = update.operator_note

        if update.confirmed and disc.status == "review_required":
            disc.status = "auto"
            disc.review_required = False

    corrected_count = sum(1 for disc in record.analysis.results if disc.status == "corrected")
    failed_count = sum(1 for disc in record.analysis.results if disc.status == "failed")
    review_required_count = sum(1 for disc in record.analysis.results if disc.review_required)
    auto_count = sum(1 for disc in record.analysis.results if disc.status == "auto")

    record.analysis.summary.corrected_count = corrected_count
    record.analysis.summary.failed_count = failed_count
    record.analysis.summary.review_required_count = review_required_count
    record.analysis.summary.auto_count = auto_count
    record.analysis.status = "REQUIRES_MANUAL_REVIEW" if review_required_count or failed_count else "VALID"

    record.operator_id = payload.operator_id
    record.updated_at = datetime.now(timezone.utc)
    _write_json(_record_path(analysis_id), _model_dump(record))
    _write_json(_result_path(analysis_id), _model_dump(record.analysis))
    return record.analysis


def build_csv_export(analysis: AnalysisResponse) -> str:
    header = [
        "row",
        "disc_id",
        "code",
        "auto_mm",
        "corrected_mm",
        "final_mm",
        "label_confidence",
        "measurement_confidence",
        "overall_confidence",
        "status",
        "source",
        "review_required",
        "no_zone_fallback_used",
        "warnings",
    ]
    rows = [",".join(header)]
    for index, disc in enumerate(analysis.results, start=1):
        warnings = " | ".join(disc.warnings).replace(",", ";")
        row = [
            str(index),
            disc.disc_id,
            disc.final_code,
            f"{disc.auto_diameter_mm:.2f}",
            "" if disc.corrected_diameter_mm is None else f"{disc.corrected_diameter_mm:.2f}",
            f"{disc.final_diameter_mm:.2f}",
            f"{disc.label_confidence:.3f}",
            f"{disc.measurement_confidence:.3f}",
            f"{disc.overall_confidence:.3f}",
            disc.status,
            disc.source,
            str(disc.review_required),
            str(disc.no_zone_fallback_used),
            warnings,
        ]
        rows.append(",".join(row))
    return "\n".join(rows)
