from __future__ import annotations

import base64
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

from app.core.config import settings
from app.evaluation.dataset import DatasetImageRecord, GroundTruthRow, index_test_images
from app.models.schemas import AnalysisResponse, DiscMeasurement
from app.services.ast_processor import run_ast_analysis
from app.utils.disc_detector import detect_discs_with_debug
from app.utils.disc_roi import extract_disc_roi
from app.utils.image_loader import load_image_from_bytes
from app.utils.plate_extractor import extract_plate
from app.utils.preprocessing import preprocess_plate
from app.utils.zone_detector import build_disc_assignment_masks, measure_zone_with_debug

EVAL_CODE_NORMALIZATION = {
    "AMP": "AM",
    "AM10": "AM",
    "GEN": "CN",
    "GN": "CN",
    "AMK": "AK",
    "TZP": "TPZ",
    "NIT": "F100",
    "FUR": "F100",
    "FOS": "FF",
    "TOP": "TOB",
}


def _model_dump(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _resize_for_analysis(image: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if settings.ANALYSIS_MAX_DIMENSION <= 0 or longest_side <= settings.ANALYSIS_MAX_DIMENSION:
        return image, 1.0

    scale = settings.ANALYSIS_MAX_DIMENSION / float(longest_side)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _save_image(path: Path, image: np.ndarray | None) -> None:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _decode_base64_image(value: Optional[str]) -> Optional[np.ndarray]:
    if not value:
        return None
    try:
        decoded = base64.b64decode(value)
    except (TypeError, ValueError):
        return None
    return cv2.imdecode(np.frombuffer(decoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def _normalize_code(code: str) -> str:
    normalized = (code or "").strip().upper()
    return EVAL_CODE_NORMALIZATION.get(normalized, normalized)


def _build_plate_overlay(image: np.ndarray, results: Iterable[DiscMeasurement]) -> np.ndarray:
    overlay = image.copy()
    for result in results:
        center = (int(round(result.center.x)), int(round(result.center.y)))
        cv2.circle(overlay, center, int(round(result.disc_radius_px)), (0, 255, 0), 2)
        cv2.circle(overlay, center, int(round(result.auto_diameter_px / 2.0)), (255, 196, 0), 2)
        cv2.putText(
            overlay,
            result.final_code,
            (center[0] - 20, max(18, center[1] - int(result.disc_radius_px) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def _build_disc_overlay(image: np.ndarray, discs: list[tuple[float, float, float]]) -> np.ndarray:
    overlay = image.copy()
    for index, (x, y, radius) in enumerate(discs, start=1):
        center = (int(round(x)), int(round(y)))
        cv2.circle(overlay, center, int(round(radius)), (0, 255, 0), 2)
        cv2.putText(
            overlay,
            f"D{index}",
            (center[0] - 16, max(18, center[1] - int(radius) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def _ground_truth_lookup(rows: List[GroundTruthRow]) -> dict[str, list[GroundTruthRow]]:
    lookup: dict[str, list[GroundTruthRow]] = defaultdict(list)
    for row in rows:
        lookup[_normalize_code(row.code)].append(row)
    return lookup


def _predicted_lookup(results: List[DiscMeasurement]) -> dict[str, list[DiscMeasurement]]:
    lookup: dict[str, list[DiscMeasurement]] = defaultdict(list)
    for result in results:
        lookup[_normalize_code(result.final_code)].append(result)
    return lookup


def _pair_measurements(
    ground_truth_rows: List[GroundTruthRow],
    results: List[DiscMeasurement],
) -> tuple[list[dict], dict]:
    ground_truth = _ground_truth_lookup(ground_truth_rows)
    predicted = _predicted_lookup(results)
    matched: list[dict] = []
    missing_codes: list[str] = []
    unexpected_codes: list[str] = []

    all_codes = sorted(set(ground_truth) | set(predicted))
    for code in all_codes:
        expected_rows = ground_truth.get(code, [])
        predicted_rows = sorted(
            predicted.get(code, []),
            key=lambda item: item.overall_confidence,
            reverse=True,
        )
        pair_count = min(len(expected_rows), len(predicted_rows))
        for index in range(pair_count):
            gt = expected_rows[index]
            predicted_row = predicted_rows[index]
            if gt.diameter_mm is None:
                continue
            error = round(predicted_row.final_diameter_mm - gt.diameter_mm, 2)
            matched.append(
                {
                    "code": code,
                    "ground_truth_code": gt.code,
                    "predicted_code": predicted_row.final_code,
                    "ground_truth_mm": gt.diameter_mm,
                    "predicted_mm": round(predicted_row.final_diameter_mm, 2),
                    "error_mm": error,
                    "abs_error_mm": round(abs(error), 2),
                    "label_confidence": round(predicted_row.label_confidence, 3),
                    "measurement_confidence": round(predicted_row.measurement_confidence, 3),
                    "review_required": predicted_row.review_required,
                    "no_zone_fallback_used": predicted_row.no_zone_fallback_used,
                }
            )
        if len(expected_rows) > pair_count:
            missing_codes.extend([code] * (len(expected_rows) - pair_count))
        if len(predicted_rows) > pair_count:
            unexpected_codes.extend([code] * (len(predicted_rows) - pair_count))

    abs_errors = [row["abs_error_mm"] for row in matched]
    comparison = {
        "expected_disc_count": len(ground_truth_rows),
        "predicted_disc_count": len(results),
        "count_matches_ground_truth": len(results) == len(ground_truth_rows),
        "matched_codes": sorted({row["code"] for row in matched}),
        "missing_codes": missing_codes,
        "unexpected_codes": unexpected_codes,
        "measurement_mae_mm": round(mean(abs_errors), 2) if abs_errors else None,
        "measurement_median_abs_error_mm": round(median(abs_errors), 2) if abs_errors else None,
        "measurement_max_abs_error_mm": round(max(abs_errors), 2) if abs_errors else None,
        "matched_disc_count": len(matched),
    }
    return matched, comparison


def _likely_confusion_pairs(
    missing_codes: List[str],
    unexpected_codes: List[str],
) -> List[dict]:
    pairs: List[dict] = []
    remaining_missing = list(missing_codes)
    for predicted in unexpected_codes:
        if not remaining_missing:
            break
        best = min(
            remaining_missing,
            key=lambda candidate: (
                abs(len(candidate) - len(predicted)),
                0 if candidate[0:1] == predicted[0:1] else 1,
                0 if candidate[-1:] == predicted[-1:] else 1,
                candidate,
            ),
        )
        pairs.append({"predicted_code": predicted, "likely_expected_code": best})
        remaining_missing.remove(best)
    return pairs


def _classify_failures(
    record: DatasetImageRecord,
    analysis: AnalysisResponse,
    comparison: Optional[dict],
    disc_debug_rows: List[dict],
) -> List[str]:
    reasons: List[str] = []

    if analysis.plate_detection is None:
        reasons.append("plate_detection_missing")
    elif record.provenance == "dryad_original" and analysis.plate_detection.shape != "rectangle":
        reasons.append("plate_detection_crop_issue")

    if comparison is not None:
        if comparison["predicted_disc_count"] < comparison["expected_disc_count"]:
            reasons.append("missed_discs")
        elif comparison["predicted_disc_count"] > comparison["expected_disc_count"]:
            reasons.append("false_positive_discs")

        if comparison["missing_codes"] or comparison["unexpected_codes"]:
            reasons.append("ocr_wrong_code")

        if any(result.final_code == "UNKNOWN" for result in analysis.results):
            reasons.append("ocr_blank_or_unknown")

        if any(
            row.get("expected_gt_mm") and row.get("result", {}).get("no_zone_fallback_used") and row["expected_gt_mm"] > 8.0
            for row in disc_debug_rows
        ):
            reasons.append("incorrect_no_zone_fallback")

        mae = comparison.get("measurement_mae_mm")
        if mae is not None and mae > 4.0:
            reasons.append("zone_measurement_error")

    if analysis.quality_report.review_required and analysis.quality_report.warnings:
        reasons.append("image_quality_or_thresholding_issue")

    radial_gap_rows = 0
    for row in disc_debug_rows:
        zone_debug = row.get("zone_debug", {})
        result = row.get("result", {})
        radial_diameter = float(zone_debug.get("radial_diameter_px", 0.0) or 0.0)
        final_diameter_px = float(result.get("auto_diameter_px", 0.0) or 0.0)
        if radial_diameter > 0 and final_diameter_px > 0 and radial_diameter >= final_diameter_px * 1.35:
            radial_gap_rows += 1
    if radial_gap_rows >= max(2, len(disc_debug_rows) // 4):
        reasons.append("zone_selection_underestimate")

    calibration_spread = float(analysis.calibration.spread_px or 0.0)
    average_disc_diameter_px = float(analysis.calibration.average_disc_diameter_px or 0.0)
    if average_disc_diameter_px > 0 and calibration_spread > average_disc_diameter_px * 0.22:
        reasons.append("scale_or_disc_radius_instability")

    if not reasons and analysis.status != "VALID":
        reasons.append("manual_review_required")
    return sorted(dict.fromkeys(reasons))


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _evaluate_record(
    record: DatasetImageRecord,
    output_dir: Path,
    *,
    strict_mode: bool,
) -> tuple[dict, List[dict], List[dict]]:
    image_path = Path(record.image_path)
    image_bytes = image_path.read_bytes()
    stage_timings: Dict[str, float] = {}
    analysis = run_ast_analysis(
        image_bytes=image_bytes,
        analysis_id=record.image_id,
        image_filename=image_path.name,
        include_debug_artifacts=False,
        stage_timings=stage_timings,
    )

    image = load_image_from_bytes(image_bytes)
    image, processing_scale = _resize_for_analysis(image)
    plate_image, plate_detection = extract_plate(image)
    preprocessed = preprocess_plate(plate_image)
    discs, disc_debug = detect_discs_with_debug(plate_image, plate_detection=plate_detection)
    zone_masks = None
    if plate_detection and plate_detection.shape == "rectangle" and len(discs) >= 8:
        zone_masks = build_disc_assignment_masks(plate_image.shape, discs, plate_detection=plate_detection)

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    discs_dir = output_dir / "discs"
    _save_image(images_dir / "input.png", image)
    _save_image(images_dir / "plate_crop.png", plate_image)
    _save_image(images_dir / "preprocessed.png", preprocessed)
    _save_image(images_dir / "disc_overlay.png", _build_disc_overlay(plate_image, discs))
    _save_image(images_dir / "final_annotated.png", _build_plate_overlay(plate_image, analysis.results))
    if record.measured_reference_path:
        reference = cv2.imread(record.measured_reference_path)
        _save_image(images_dir / "measured_reference.png", reference)
    _save_image(images_dir / "plate_mask.png", disc_debug.get("plate_mask"))
    _save_image(images_dir / "white_mask.png", disc_debug.get("white_mask"))
    _save_image(images_dir / "core_mask.png", disc_debug.get("core_mask"))

    disc_debug_rows: List[dict] = []
    for index, result in enumerate(analysis.results):
        disc_entry_dir = discs_dir / result.disc_id
        disc_entry_dir.mkdir(parents=True, exist_ok=True)
        crop = extract_disc_roi(plate_image, result.center.x, result.center.y, result.disc_radius_px, expand_ratio=0.24, mask_scale=0.82)
        relaxed_crop = extract_disc_roi(plate_image, result.center.x, result.center.y, result.disc_radius_px, expand_ratio=0.30, mask_scale=0.98)
        _save_image(disc_entry_dir / "ocr_crop.png", crop)
        _save_image(disc_entry_dir / "ocr_crop_relaxed.png", relaxed_crop)

        zone_debug = {}
        if index < len(discs):
            zone_result, zone_debug = measure_zone_with_debug(
                plate_image,
                discs[index],
                plate_detection=plate_detection,
                extra_valid_mask=zone_masks[index] if zone_masks and index < len(zone_masks) else None,
            )
            _save_image(disc_entry_dir / "zone_roi.png", zone_debug.get("roi"))
            _save_image(disc_entry_dir / "zone_contrast.png", zone_debug.get("contrast"))
            _save_image(disc_entry_dir / "zone_edges.png", zone_debug.get("edge_map"))
            _save_image(disc_entry_dir / "zone_mask.png", zone_debug.get("zone_mask"))
            _save_image(disc_entry_dir / "zone_valid_mask.png", zone_debug.get("valid_mask"))
            _save_image(disc_entry_dir / "zone_overlay.png", zone_debug.get("overlay"))
        overlay_image = _decode_base64_image(result.overlay_image_base64)
        crop_image = _decode_base64_image(result.crop_image_base64)
        _save_image(disc_entry_dir / "overlay_from_result.png", overlay_image)
        _save_image(disc_entry_dir / "crop_from_result.png", crop_image)

        expected_gt_mm = None
        disc_debug_rows.append(
            {
                "disc_id": result.disc_id,
                "result": _model_dump(result),
                "zone_debug": {
                    key: value
                    for key, value in zone_debug.items()
                    if not isinstance(value, np.ndarray)
                },
                "expected_gt_mm": expected_gt_mm,
            }
        )

    matched_measurements: List[dict] = []
    comparison = None
    ocr_confusions: List[dict] = []
    if record.ground_truth_rows:
        matched_measurements, comparison = _pair_measurements(record.ground_truth_rows, analysis.results)
        confusion_pairs = _likely_confusion_pairs(
            comparison["missing_codes"],
            comparison["unexpected_codes"],
        )
        for pair in confusion_pairs:
            matching_result = next(
                (item for item in analysis.results if _normalize_code(item.final_code) == pair["predicted_code"]),
                None,
            )
            ocr_confusions.append(
                {
                    "image_id": record.image_id,
                    "predicted_code": pair["predicted_code"],
                    "likely_expected_code": pair["likely_expected_code"],
                    "raw_ocr_text": matching_result.raw_ocr_text if matching_result else "",
                    "normalized_ocr_text": matching_result.normalized_ocr_text if matching_result else "",
                    "candidates": " | ".join(matching_result.label_candidates[:5]) if matching_result else "",
                }
            )

        gt_lookup = _ground_truth_lookup(record.ground_truth_rows)
        for row in disc_debug_rows:
            norm_code = _normalize_code(row["result"].get("final_code", ""))
            if gt_lookup.get(norm_code):
                row["expected_gt_mm"] = gt_lookup[norm_code][0].diameter_mm

    root_causes = _classify_failures(record, analysis, comparison, disc_debug_rows)
    strict_failed = strict_mode and (
        analysis.status != "VALID" or any(result.review_required for result in analysis.results)
    )
    full_pass = False
    if comparison is None:
        full_pass = analysis.status == "VALID" and not strict_failed
    else:
        mae = comparison.get("measurement_mae_mm")
        full_pass = (
            comparison["count_matches_ground_truth"]
            and not comparison["missing_codes"]
            and not comparison["unexpected_codes"]
            and (mae is None or mae <= 3.0)
            and not strict_failed
        )

    debug_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "record": record.to_dict(),
        "processing_scale": processing_scale,
        "stage_timings": {key: round(float(value), 4) for key, value in stage_timings.items()},
        "plate_detection": _model_dump(analysis.plate_detection) if analysis.plate_detection else None,
        "quality_report": _model_dump(analysis.quality_report),
        "calibration": _model_dump(analysis.calibration),
        "comparison": comparison,
        "root_causes": root_causes,
        "disc_debug": disc_debug_rows,
        "disc_detection_debug": {
            "component_candidate_count": len(disc_debug.get("component_candidates", [])),
            "hough_candidate_count": len(disc_debug.get("hough_candidates", [])),
            "radius_bounds": disc_debug.get("radius_bounds", {}),
        },
    }

    _write_json(output_dir / "result.json", _model_dump(analysis))
    _write_json(output_dir / "debug.json", debug_payload)

    summary_row = {
        "image_id": record.image_id,
        "source_group": record.source_group,
        "provenance": record.provenance,
        "analysis_status": analysis.status,
        "predicted_disc_count": len(analysis.results),
        "expected_disc_count": comparison["expected_disc_count"] if comparison else "",
        "count_matches_ground_truth": comparison["count_matches_ground_truth"] if comparison else "",
        "matched_disc_count": comparison["matched_disc_count"] if comparison else "",
        "measurement_mae_mm": comparison["measurement_mae_mm"] if comparison else "",
        "measurement_max_abs_error_mm": comparison["measurement_max_abs_error_mm"] if comparison else "",
        "missing_codes": " | ".join(comparison["missing_codes"]) if comparison else "",
        "unexpected_codes": " | ".join(comparison["unexpected_codes"]) if comparison else "",
        "review_required_count": sum(1 for result in analysis.results if result.review_required),
        "failed_count": sum(1 for result in analysis.results if result.status == "failed"),
        "strict_failed": strict_failed,
        "full_pass": full_pass,
        "root_causes": " | ".join(root_causes),
        "warnings": " | ".join(analysis.warnings + analysis.quality_report.warnings),
    }
    return summary_row, matched_measurements, ocr_confusions


def run_corpus_evaluation(
    *,
    test_root: Path,
    output_root: Path,
    sources: Optional[set[str]] = None,
    limit: Optional[int] = None,
    strict_mode: bool = False,
) -> dict:
    records = index_test_images(test_root)
    manifest = [record.to_dict() for record in records]

    run_root = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_root / "manifest.json", {"records": manifest})

    analyzable_records = [
        record
        for record in records
        if record.analyzable and (not sources or record.source_group in sources or record.provenance in sources)
    ]
    if limit is not None:
        analyzable_records = analyzable_records[:limit]

    per_image_rows: List[dict] = []
    measurement_rows: List[dict] = []
    ocr_rows: List[dict] = []
    root_cause_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter(record.source_group for record in analyzable_records)

    for index, record in enumerate(analyzable_records, start=1):
        safe_name = f"{record.source_group}__{record.image_id}"
        image_output_dir = run_root / "per_image" / safe_name
        summary_row, matched_measurements, ocr_confusions = _evaluate_record(
            record,
            image_output_dir,
            strict_mode=strict_mode,
        )
        per_image_rows.append(summary_row)
        measurement_rows.extend(
            [{"image_id": record.image_id, **row} for row in matched_measurements]
        )
        ocr_rows.extend(ocr_confusions)
        for cause in filter(None, summary_row["root_causes"].split(" | ")):
            root_cause_counter[cause] += 1
        print(f"[{index}/{len(analyzable_records)}] Evaluated {record.image_id}")

    aggregate = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluated_image_count": len(per_image_rows),
        "images_with_ground_truth": sum(1 for row in per_image_rows if row["expected_disc_count"] != ""),
        "source_counts": dict(source_counter),
        "full_pass_count": sum(1 for row in per_image_rows if row["full_pass"]),
        "strict_fail_count": sum(1 for row in per_image_rows if row["strict_failed"]),
        "analysis_status_breakdown": dict(Counter(row["analysis_status"] for row in per_image_rows)),
        "measurement_mae_mm": round(mean([row["measurement_mae_mm"] for row in per_image_rows if row["measurement_mae_mm"] != ""]), 2)
        if any(row["measurement_mae_mm"] != "" for row in per_image_rows)
        else None,
        "measurement_max_abs_error_mm": max(
            (float(row["measurement_max_abs_error_mm"]) for row in per_image_rows if row["measurement_max_abs_error_mm"] != ""),
            default=None,
        ),
        "root_causes": dict(root_cause_counter),
    }

    _write_json(run_root / "summary.json", aggregate)
    _write_json(run_root / "root_cause_summary.json", dict(root_cause_counter))

    if per_image_rows:
        _write_csv(
            run_root / "per_image_summary.csv",
            per_image_rows,
            list(per_image_rows[0].keys()),
        )
    if measurement_rows:
        _write_csv(
            run_root / "measurement_errors.csv",
            measurement_rows,
            list(measurement_rows[0].keys()),
        )
    if ocr_rows:
        _write_csv(
            run_root / "ocr_confusions.csv",
            ocr_rows,
            list(ocr_rows[0].keys()),
        )

    return {"run_root": str(run_root), "summary": aggregate}
