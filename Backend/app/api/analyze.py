from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from app.models.schemas import AnalysisResponse, SaveReviewRequest
from app.services.analysis_store import apply_review_update, build_csv_export, load_analysis_record
from app.services.ast_processor import process_ast_image

router = APIRouter()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    image: UploadFile = File(...),
    include_debug_artifacts: bool = Form(False),
) -> AnalysisResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    image_bytes = await image.read()
    return await process_ast_image(
        image_bytes=image_bytes,
        image_filename=image.filename,
        include_debug_artifacts=include_debug_artifacts,
    )


@router.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: str) -> AnalysisResponse:
    return load_analysis_record(analysis_id).analysis


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
