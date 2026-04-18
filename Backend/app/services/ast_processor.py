from __future__ import annotations

import logging
import uuid
from time import perf_counter
from typing import Callable, Dict, Optional

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

ProgressCallback = Callable[[float, str, str, Optional[Dict[str, float]]], None]


def run_ast_analysis(
    image_bytes: bytes,
    *,
    analysis_id: str | None = None,
    image_filename: str | None = None,
    include_debug_artifacts: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
    stage_timings: Optional[Dict[str, float]] = None,
) -> AnalysisResponse:
    timings = stage_timings if stage_timings is not None else {}

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    if len(image_bytes) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )

    analysis_id = analysis_id or str(uuid.uuid4())
    total_started = perf_counter()

    if progress_callback:
        progress_callback(0.08, "image_decode", "Loading the uploaded image.", timings.copy())

    started = perf_counter()
    try:
        image = load_image_from_bytes(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    timings["image_decode_seconds"] = perf_counter() - started

    if progress_callback:
        progress_callback(0.18, "plate_extraction", "Detecting the plate area.", timings.copy())

    started = perf_counter()
    plate_image, plate_detection = extract_plate(image)
    timings["plate_extraction_seconds"] = perf_counter() - started

    if progress_callback:
        progress_callback(0.28, "quality_validation", "Checking image quality.", timings.copy())

    started = perf_counter()
    quality_report = validate_image_quality(plate_image, plate_detection=plate_detection)
    timings["quality_validation_seconds"] = perf_counter() - started

    if progress_callback:
        progress_callback(
            0.32,
            "pipeline",
            "Running disc detection, OCR, and zone measurement.",
            timings.copy(),
        )

    started = perf_counter()
    analysis = hybrid_analysis_pipeline(
        image=plate_image,
        analysis_id=analysis_id,
        plate_detection=plate_detection,
        quality_report=quality_report,
        image_filename=image_filename,
        include_debug_artifacts=include_debug_artifacts,
        progress_callback=progress_callback,
        stage_timings=timings,
    )
    timings["pipeline_seconds"] = perf_counter() - started
    timings["total_runtime_seconds"] = perf_counter() - total_started

    logger.info(
        "Analysis %s completed with %s detected discs. Stage timings: %s",
        analysis.analysis_id,
        len(analysis.results),
        {key: round(value, 3) for key, value in timings.items()},
    )
    return analysis


async def process_ast_image(
    image_bytes: bytes,
    image_filename: str | None = None,
    include_debug_artifacts: bool = False,
) -> AnalysisResponse:
    stage_timings: Dict[str, float] = {}
    analysis = run_ast_analysis(
        image_bytes=image_bytes,
        image_filename=image_filename,
        include_debug_artifacts=include_debug_artifacts,
        stage_timings=stage_timings,
    )
    return save_analysis_record(
        analysis,
        image_bytes=image_bytes,
        image_filename=image_filename,
        stage_timings=stage_timings,
    )
