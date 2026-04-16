from __future__ import annotations

import logging
import uuid

from fastapi import HTTPException

from app.core.config import settings
from app.ml.inference import hybrid_analysis_pipeline
from app.models.schemas import AnalysisResponse
from app.services.analysis_store import save_analysis_record
from app.services.quality_validator import validate_image_quality
from app.utils.image_loader import load_image_from_bytes
from app.utils.plate_extractor import extract_plate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def process_ast_image(
    image_bytes: bytes,
    image_filename: str | None = None,
    include_debug_artifacts: bool = False,
) -> AnalysisResponse:
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    if len(image_bytes) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )

    try:
        image = load_image_from_bytes(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    plate_image, plate_detection = extract_plate(image)
    quality_report = validate_image_quality(plate_image, plate_detection=plate_detection)

    analysis = hybrid_analysis_pipeline(
        image=plate_image,
        analysis_id=str(uuid.uuid4()),
        plate_detection=plate_detection,
        quality_report=quality_report,
        image_filename=image_filename,
        include_debug_artifacts=include_debug_artifacts,
    )

    logger.info("Analysis %s completed with %s detected discs.", analysis.analysis_id, len(analysis.results))
    return save_analysis_record(analysis, image_bytes=image_bytes, image_filename=image_filename)
