from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import cv2
import numpy as np
from fastapi import HTTPException

from app.core.config import settings
from app.models.schemas import AnalysisResponse, PersistedAnalysisRecord, SaveReviewRequest
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


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


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


def save_analysis_record(
    analysis: AnalysisResponse,
    image_bytes: bytes,
    image_filename: str | None = None,
) -> AnalysisResponse:
    directory = _analysis_dir(analysis.analysis_id)
    directory.mkdir(parents=True, exist_ok=True)

    image_path = directory / "original_upload.bin"
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
    return analysis


def load_analysis_record(analysis_id: str) -> PersistedAnalysisRecord:
    path = _record_path(analysis_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Analysis record not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    return PersistedAnalysisRecord(**data)


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
