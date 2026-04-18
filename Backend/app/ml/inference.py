from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone
from time import perf_counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
from app.services.panel_resolver import apply_standard_panel_resolution
from app.utils.calibration import build_calibration, pixels_to_mm
from app.utils.disc_detector import detect_discs_with_debug
from app.utils.disc_roi import extract_disc_roi
from app.utils.measurement import apply_no_zone_rule, calculate_inhibition_result
from app.utils.zone_detector import build_disc_assignment_masks, measure_zone

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[float, str, str, Optional[Dict[str, float]]], None]


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


def _build_plate_detection_overlay(image: np.ndarray, plate_detection: Optional[PlateDetection]) -> np.ndarray:
    overlay = image.copy()
    if plate_detection is None:
        return overlay

    if plate_detection.shape == "rectangle" and plate_detection.bounding_box is not None:
        box = plate_detection.bounding_box
        cv2.rectangle(
            overlay,
            (int(round(box.x1)), int(round(box.y1))),
            (int(round(box.x2)), int(round(box.y2))),
            (255, 255, 0),
            3,
        )
    else:
        center = (int(round(plate_detection.center.x)), int(round(plate_detection.center.y)))
        cv2.circle(overlay, center, int(round(plate_detection.radius_px)), (255, 255, 0), 3)
    return overlay


def _build_disc_detection_overlay(
    image: np.ndarray,
    disc_detection_debug: Dict[str, object],
) -> np.ndarray:
    overlay = image.copy()
    for record in disc_detection_debug.get("raw_candidate_records", []):
        center = (int(round(record["x"])), int(round(record["y"])))
        cv2.circle(overlay, center, int(round(record["radius"])), (0, 220, 255), 1)

    for record in disc_detection_debug.get("rejected_candidates", []):
        center = (int(round(record["x"])), int(round(record["y"])))
        cv2.circle(overlay, center, int(round(record["radius"])), (0, 0, 255), 1)
        reason = str(record.get("reason", "reject"))[:18]
        cv2.putText(
            overlay,
            reason,
            (center[0] - 18, max(14, center[1] - int(round(record["radius"])) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    for index, record in enumerate(disc_detection_debug.get("accepted_candidates", []), start=1):
        center = (int(round(record["x"])), int(round(record["y"])))
        cv2.circle(overlay, center, int(round(record["radius"])), (0, 255, 0), 2)
        cv2.putText(
            overlay,
            f"D{index}",
            (center[0] - 14, max(16, center[1] - int(round(record["radius"])) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
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
    progress_callback: Optional[ProgressCallback] = None,
    stage_timings: Optional[Dict[str, float]] = None,
) -> AnalysisResponse:
    timings = stage_timings if stage_timings is not None else {}

    started = perf_counter()
    discs, disc_detection_debug = detect_discs_with_debug(image, plate_detection=plate_detection)
    timings["disc_detection_seconds"] = perf_counter() - started
    if progress_callback:
        progress_callback(
            0.4,
            "disc_detection",
            f"Detected {len(discs)} disc(s).",
            timings.copy(),
        )

    started = perf_counter()
    calibration = build_calibration(discs)
    timings["calibration_seconds"] = perf_counter() - started

    warnings = list(quality_report.warnings) + list(calibration.warnings)
    results: List[DiscMeasurement] = []
    ocr_debug_entries: List[Dict[str, object]] = []

    if not discs:
        debug_artifacts: Dict[str, object] = {}
        if include_debug_artifacts:
            debug_artifacts = {
                "plate_detection_overlay_base64": _encode_png_base64(_build_plate_detection_overlay(image, plate_detection)),
                "disc_detection_overlay_base64": _encode_png_base64(
                    _build_disc_detection_overlay(image, disc_detection_debug)
                ),
                "disc_detection_debug": {
                    "radius_bounds": disc_detection_debug.get("radius_bounds", {}),
                    "fallback_detection_used": bool(disc_detection_debug.get("fallback_detection_used", False)),
                    "suspicious_low_count": bool(disc_detection_debug.get("suspicious_low_count", False)),
                    "raw_candidate_records": disc_detection_debug.get("raw_candidate_records", []),
                    "accepted_candidates": disc_detection_debug.get("accepted_candidates", []),
                    "rejected_candidates": disc_detection_debug.get("rejected_candidates", []),
                },
            }
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
            debug_artifacts=debug_artifacts,
        )

    measurement_total = 0.0
    ocr_total = 0.0
    overlay_total = 0.0
    total_discs = len(discs)
    dense_zone_masks: List[np.ndarray] | None = None
    if plate_detection and plate_detection.shape == "rectangle" and total_discs >= 8:
        dense_zone_masks = build_disc_assignment_masks(image.shape, discs, plate_detection=plate_detection)

    for index, disc in enumerate(discs, start=1):
        x, y, radius = disc
        disc_diameter_px = round(radius * 2.0, 2)

        measurement_started = perf_counter()
        zone_result = measure_zone(
            image,
            disc,
            plate_detection=plate_detection,
            extra_valid_mask=dense_zone_masks[index - 1] if dense_zone_masks and (index - 1) < len(dense_zone_masks) else None,
        )
        measurement_total += perf_counter() - measurement_started
        auto_diameter_px = float(zone_result["diameter_px"])
        auto_diameter_mm = calculate_inhibition_result(
            zone_diameter_px=auto_diameter_px,
            disc_diameter_px=disc_diameter_px,
            mm_per_pixel=calibration.mm_per_pixel,
            no_zone_detected=bool(zone_result["no_zone_fallback_used"]),
        )
        if bool(zone_result["no_zone_fallback_used"]):
            auto_diameter_mm = apply_no_zone_rule(auto_diameter_mm)

        ocr_started = perf_counter()
        crop = extract_disc_roi(image, x, y, radius, expand_ratio=0.24, mask_scale=0.82)
        relaxed_crop = extract_disc_roi(image, x, y, radius, expand_ratio=0.3, mask_scale=0.98)
        label_prediction = predict_disc_class_ocr(crop, include_debug=include_debug_artifacts)
        if (
            (
                str(label_prediction.get("confidence_tier", "")) != "high_confidence_exact"
                and float(label_prediction.get("confidence", 0.0)) < 0.97
            )
            or str(label_prediction.get("code", "UNKNOWN")).upper() == "UNKNOWN"
            or len(str(label_prediction.get("code", ""))) <= 1
        ):
            label_prediction = predict_disc_class_ocr(
                crop,
                fallback_images=[relaxed_crop],
                include_debug=include_debug_artifacts,
            )
        ocr_total += perf_counter() - ocr_started
        detected_code = str(label_prediction.get("code", "UNKNOWN")).upper()
        label_confidence = float(label_prediction.get("confidence", 0.0))
        label_candidates = [candidate.upper() for candidate in label_prediction.get("candidates", [])]
        label_tier = str(label_prediction.get("confidence_tier", "failed_unknown"))

        result_warnings: List[str] = []
        zone_warning = str(zone_result.get("warning", "") or "")
        if zone_warning:
            result_warnings.append(zone_warning)
        if label_tier == "failed_unknown":
            result_warnings.append("Disc label could not be confirmed automatically.")
        elif label_tier == "uncertain_manual_confirmation_required":
            result_warnings.append("Disc label needs manual confirmation before the result is accepted.")
        elif label_tier == "probable_match":
            result_warnings.append("Disc label is a probable whitelist match; confirm it before saving.")

        measurement_confidence = float(zone_result["confidence"])
        overall_confidence = round((label_confidence * 0.45) + (measurement_confidence * 0.55), 3)
        review_required = bool(zone_result["review_required"]) or label_tier != "high_confidence_exact" or quality_report.review_required

        status = "auto"
        if detected_code == "UNKNOWN" and measurement_confidence < 0.35:
            status = "failed"
            review_required = True
        elif review_required:
            status = "review_required"

        overlay_started = perf_counter()
        overlay_image = _build_disc_overlay(image, disc, auto_diameter_px, detected_code if detected_code != "UNKNOWN" else f"D{index}")
        overlay_total += perf_counter() - overlay_started
        if include_debug_artifacts:
            ocr_debug_entries.append(
                {
                    "disc_index": index,
                    "disc_center": {"x": round(x, 2), "y": round(y, 2)},
                    "selected_code": detected_code,
                    "confidence_tier": label_tier,
                    "ocr_debug": label_prediction.get("ocr_debug", {}),
                }
            )
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
                label_confidence_tier=label_tier,
                label_candidates=label_candidates,
                whitelist_candidates_considered=[
                    candidate.upper() for candidate in label_prediction.get("whitelist_candidates", [])
                ],
                label_engine=str(label_prediction.get("engine", "hybrid-ocr-whitelist")),
                label_decision_source=str(label_prediction.get("decision_source", "ocr")),
                label_selection_reason=str(label_prediction.get("selection_reason", "")),
                raw_ocr_text=str(label_prediction.get("raw_ocr_text", "")),
                normalized_ocr_text=str(label_prediction.get("normalized_text", "")),
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
        if progress_callback:
            progress_callback(
                0.45 + ((index / float(total_discs)) * 0.4),
                "disc_analysis",
                f"Processed disc {index} of {total_discs}.",
                timings.copy(),
            )

    results = sorted(results, key=lambda result: (result.center.y, result.center.x))
    for index, result in enumerate(results, start=1):
        result.index = index
        result.disc_id = f"disc-{index}"

    timings["measurement_seconds"] = measurement_total
    timings["ocr_seconds"] = ocr_total
    timings["overlay_build_seconds"] = overlay_total

    started = perf_counter()
    apply_standard_panel_resolution(results, image.shape)
    timings["panel_resolution_seconds"] = perf_counter() - started

    summary = _summarize(results)

    if progress_callback:
        progress_callback(
            0.92,
            "result_serialization",
            "Finalizing analysis result.",
            timings.copy(),
        )

    started = perf_counter()
    plate_overlay = _build_plate_overlay(image, results)
    debug_artifacts: Dict[str, object] = {}
    if include_debug_artifacts:
        debug_artifacts = {
            "plate_overlay_base64": _encode_png_base64(plate_overlay),
            "plate_detection_overlay_base64": _encode_png_base64(_build_plate_detection_overlay(image, plate_detection)),
            "disc_detection_overlay_base64": _encode_png_base64(
                _build_disc_detection_overlay(image, disc_detection_debug)
            ),
            "ocr_debug": ocr_debug_entries,
            "calibration_summary": {
                "disc_count": calibration.disc_count,
                "average_disc_diameter_px": calibration.average_disc_diameter_px,
                "mm_per_pixel": calibration.mm_per_pixel,
            },
            "disc_detection_debug": {
                "radius_bounds": disc_detection_debug.get("radius_bounds", {}),
                "fallback_detection_used": bool(disc_detection_debug.get("fallback_detection_used", False)),
                "suspicious_low_count": bool(disc_detection_debug.get("suspicious_low_count", False)),
                "raw_candidate_records": disc_detection_debug.get("raw_candidate_records", []),
                "accepted_candidates": disc_detection_debug.get("accepted_candidates", []),
                "rejected_candidates": disc_detection_debug.get("rejected_candidates", []),
            },
        }
    timings["result_serialization_seconds"] = perf_counter() - started

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
