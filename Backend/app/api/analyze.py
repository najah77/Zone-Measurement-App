from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from app.core.config import settings
from app.models.schemas import (
    AnalysisJobStatusResponse,
    AnalysisResponse,
    AnalysisSubmissionResponse,
    SaveReviewRequest,
)
from app.services.analysis_jobs import submit_analysis_job
from app.services.analysis_store import (
    apply_review_update,
    build_csv_export,
    load_analysis_job_status,
    load_analysis_record,
    load_analysis_result,
)

router = APIRouter()


@router.post("/analyze", response_model=AnalysisSubmissionResponse, status_code=202)
async def analyze(
    image: UploadFile = File(...),
    include_debug_artifacts: bool = Form(False),
) -> AnalysisSubmissionResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(image_bytes) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )
    return submit_analysis_job(
        image_bytes=image_bytes,
        image_filename=image.filename,
        include_debug_artifacts=include_debug_artifacts,
    )


@router.get("/analyze/{analysis_id}/status", response_model=AnalysisJobStatusResponse)
async def get_analysis_status(analysis_id: str) -> AnalysisJobStatusResponse:
    return load_analysis_job_status(analysis_id)


@router.get("/analyze/{analysis_id}/result", response_model=AnalysisResponse)
async def get_analysis_result(analysis_id: str) -> AnalysisResponse:
    return load_analysis_result(analysis_id)


@router.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: str) -> AnalysisResponse:
    return load_analysis_result(analysis_id)


@router.post("/analysis/{analysis_id}/review", response_model=AnalysisResponse)
async def save_review(analysis_id: str, payload: SaveReviewRequest) -> AnalysisResponse:
    return apply_review_update(analysis_id, payload)


@router.get("/analysis/{analysis_id}/export", response_class=PlainTextResponse)
async def export_analysis(analysis_id: str) -> PlainTextResponse:
    analysis = load_analysis_record(analysis_id).analysis
    csv_body = build_csv_export(analysis)
    return PlainTextResponse(
        content=csv_body,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{analysis_id}.csv"'},
    )
