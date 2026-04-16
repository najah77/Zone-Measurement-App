from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.ml.ocr_classifier import predict_disc_class_ocr
from app.models.schemas import (
    AnalysisResponse,
    AnalysisSummary,
    DiscMeasurement,
    PlateDetection,
    Point,
    QualityReport,
)
from app.utils.calibration import build_calibration, pixels_to_mm
from app.utils.disc_detector import detect_discs
from app.utils.disc_roi import extract_disc_roi
from app.utils.measurement import apply_no_zone_rule, calculate_inhibition_result
from app.utils.zone_detector import measure_zone

logger = logging.getLogger(__name__)


def _encode_png_base64(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _build_disc_overlay(image: np.ndarray, disc: Tuple[float, float, float], zone_diameter_px: float, label: str) -> np.ndarray:
    x, y, radius = disc
    roi_radius = int(max(radius * 3.2, zone_diameter_px / 2 + 12))
    x1 = max(0, int(x - roi_radius))
    y1 = max(0, int(y - roi_radius))
    x2 = min(image.shape[1], int(x + roi_radius))
    y2 = min(image.shape[0], int(y + roi_radius))
    overlay = image[y1:y2, x1:x2].copy()
    center = (int(x - x1), int(y - y1))
    cv2.circle(overlay, center, int(round(radius)), (0, 255, 0), 2)
    cv2.circle(overlay, center, int(round(zone_diameter_px / 2.0)), (255, 196, 0), 2)
    cv2.putText(
        overlay,
        label,
        (max(4, center[0] - 18), max(16, center[1] - int(radius) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def _build_plate_overlay(image: np.ndarray, results: Sequence[DiscMeasurement]) -> np.ndarray:
    overlay = image.copy()
    for result in results:
        center = (int(result.center.x), int(result.center.y))
        cv2.circle(overlay, center, int(round(result.disc_radius_px)), (0, 255, 0), 2)
        cv2.circle(overlay, center, int(round(result.auto_diameter_px / 2.0)), (255, 196, 0), 2)
        cv2.putText(
            overlay,
            result.final_code,
            (center[0] - 18, max(18, center[1] - int(result.disc_radius_px) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def _summarize(results: Sequence[DiscMeasurement]) -> AnalysisSummary:
    review_required_count = sum(1 for result in results if result.review_required)
    corrected_count = sum(1 for result in results if result.status == "corrected")
    failed_count = sum(1 for result in results if result.status == "failed")
    auto_count = sum(1 for result in results if result.status == "auto")
    return AnalysisSummary(
        total_discs=len(results),
        review_required_count=review_required_count,
        corrected_count=corrected_count,
        failed_count=failed_count,
        auto_count=auto_count,
    )


def hybrid_analysis_pipeline(
    image: np.ndarray,
    analysis_id: str,
    plate_detection: Optional[PlateDetection],
    quality_report: QualityReport,
    image_filename: Optional[str] = None,
    include_debug_artifacts: bool = False,
) -> AnalysisResponse:
    discs = detect_discs(image)
    calibration = build_calibration(discs)

    warnings = list(quality_report.warnings) + list(calibration.warnings)
    results: List[DiscMeasurement] = []

    if not discs:
        return AnalysisResponse(
            analysis_id=analysis_id,
            status="FAILED",
            algorithm_version=settings.ALGORITHM_VERSION,
            created_at=datetime.now(timezone.utc),
            image_filename=image_filename,
            plate_detection=plate_detection,
            quality_report=quality_report,
            calibration=calibration,
            results=[],
            summary=AnalysisSummary(total_discs=0, failed_count=1),
            warnings=warnings + ["No discs were detected automatically."],
        )

    for index, disc in enumerate(discs, start=1):
        x, y, radius = disc
        disc_diameter_px = round(radius * 2.0, 2)
        zone_result = measure_zone(image, disc)
        auto_diameter_px = float(zone_result["diameter_px"])
        auto_diameter_mm = calculate_inhibition_result(
            zone_diameter_px=auto_diameter_px,
            disc_diameter_px=disc_diameter_px,
            mm_per_pixel=calibration.mm_per_pixel,
        )
        auto_diameter_mm = apply_no_zone_rule(auto_diameter_mm)

        crop = extract_disc_roi(image, x, y, radius)
        label_prediction = predict_disc_class_ocr(crop)
        detected_code = str(label_prediction.get("code", "UNKNOWN")).upper()
        label_confidence = float(label_prediction.get("confidence", 0.0))
        label_candidates = [candidate.upper() for candidate in label_prediction.get("candidates", [])]

        result_warnings: List[str] = []
        zone_warning = str(zone_result.get("warning", "") or "")
        if zone_warning:
            result_warnings.append(zone_warning)
        if detected_code == "UNKNOWN":
            result_warnings.append("Disc label could not be read automatically.")
        elif label_confidence < settings.OCR_MIN_CONFIDENCE:
            result_warnings.append("Disc label confidence is low; confirm the suggested code.")

        measurement_confidence = float(zone_result["confidence"])
        overall_confidence = round((label_confidence * 0.45) + (measurement_confidence * 0.55), 3)
        review_required = bool(zone_result["review_required"]) or label_confidence < settings.OCR_MIN_CONFIDENCE or quality_report.review_required

        status = "auto"
        if detected_code == "UNKNOWN" and measurement_confidence < 0.35:
            status = "failed"
            review_required = True
        elif review_required:
            status = "review_required"

        overlay_image = _build_disc_overlay(image, disc, auto_diameter_px, detected_code if detected_code != "UNKNOWN" else f"D{index}")
        results.append(
            DiscMeasurement(
                disc_id=f"disc-{index}",
                index=index,
                center=Point(x=round(x, 2), y=round(y, 2)),
                disc_radius_px=round(radius, 2),
                disc_diameter_px=disc_diameter_px,
                detected_code=detected_code,
                final_code=detected_code,
                label_confidence=round(label_confidence, 3),
                label_candidates=label_candidates,
                label_engine="tesseract-ocr",
                auto_diameter_px=round(auto_diameter_px, 2),
                auto_diameter_mm=round(auto_diameter_mm, 2),
                final_diameter_mm=round(auto_diameter_mm, 2),
                measurement_confidence=round(measurement_confidence, 3),
                overall_confidence=overall_confidence,
                source="auto",
                status=status,
                review_required=review_required,
                no_zone_fallback_used=bool(zone_result["no_zone_fallback_used"]),
                warnings=result_warnings,
                measurement_method=str(zone_result["method"]),
                calibration_mm_per_pixel=calibration.mm_per_pixel,
                crop_image_base64=_encode_png_base64(crop) if crop.size else None,
                overlay_image_base64=_encode_png_base64(overlay_image),
            )
        )

    results = sorted(results, key=lambda result: (result.center.y, result.center.x))
    for index, result in enumerate(results, start=1):
        result.index = index
        result.disc_id = f"disc-{index}"

    summary = _summarize(results)
    plate_overlay = _build_plate_overlay(image, results)
    debug_artifacts: Dict[str, object] = {}
    if include_debug_artifacts:
        debug_artifacts = {
            "plate_overlay_base64": _encode_png_base64(plate_overlay),
            "calibration_summary": {
                "disc_count": calibration.disc_count,
                "average_disc_diameter_px": calibration.average_disc_diameter_px,
                "mm_per_pixel": calibration.mm_per_pixel,
            },
        }

    status = "REQUIRES_MANUAL_REVIEW" if summary.review_required_count or summary.failed_count else "VALID"
    return AnalysisResponse(
        analysis_id=analysis_id,
        status=status,
        algorithm_version=settings.ALGORITHM_VERSION,
        created_at=datetime.now(timezone.utc),
        image_filename=image_filename,
        plate_detection=plate_detection,
        quality_report=quality_report,
        calibration=calibration,
        results=results,
        summary=summary,
        warnings=warnings,
        debug_artifacts=debug_artifacts,
    )
