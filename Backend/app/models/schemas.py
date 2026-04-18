from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Point(BaseModel):
    x: float
    y: float


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class PlateDetection(BaseModel):
    center: Point
    radius_px: float
    shape: str = "circle"
    bounding_box: Optional[BoundingBox] = None
    clipped: bool = False
    clip_fraction: float = 0.0
    method: str = "contour"
    warnings: List[str] = Field(default_factory=list)


class CalibrationDetails(BaseModel):
    disc_diameter_mm: float = 6.0
    average_disc_diameter_px: float = 0.0
    mm_per_pixel: float = 1.0
    disc_count: int = 0
    spread_px: float = 0.0
    method: str = "median-disc-diameter"
    warnings: List[str] = Field(default_factory=list)


class QualityReport(BaseModel):
    blur_score: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    glare_fraction: float = 0.0
    clipped_plate: bool = False
    severe_skew: bool = False
    warnings: List[str] = Field(default_factory=list)
    review_required: bool = False


class DiscMeasurement(BaseModel):
    disc_id: str
    index: int
    center: Point
    disc_radius_px: float
    disc_diameter_px: float
    disc_diameter_mm: float = 6.0
    detected_code: str
    final_code: str
    label_confidence: float = 0.0
    label_confidence_tier: str = "failed_unknown"
    label_candidates: List[str] = Field(default_factory=list)
    whitelist_candidates_considered: List[str] = Field(default_factory=list)
    label_engine: str = "ocr"
    label_decision_source: str = "ocr"
    label_selection_reason: str = ""
    raw_ocr_text: str = ""
    normalized_ocr_text: str = ""
    layout_suggestion: Optional[str] = None
    auto_diameter_px: float = 0.0
    auto_diameter_mm: float = 6.0
    corrected_diameter_mm: Optional[float] = None
    final_diameter_mm: float = 6.0
    measurement_confidence: float = 0.0
    overall_confidence: float = 0.0
    source: str = "auto"
    status: str = "auto"
    review_required: bool = False
    no_zone_fallback_used: bool = False
    warnings: List[str] = Field(default_factory=list)
    measurement_method: str = "radial-profile"
    calibration_mm_per_pixel: float = 1.0
    crop_image_base64: Optional[str] = None
    overlay_image_base64: Optional[str] = None
    operator_note: Optional[str] = None


class AnalysisSummary(BaseModel):
    total_discs: int = 0
    review_required_count: int = 0
    corrected_count: int = 0
    failed_count: int = 0
    auto_count: int = 0


class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str = "VALID"
    algorithm_version: str
    created_at: datetime
    image_filename: Optional[str] = None
    plate_detection: Optional[PlateDetection] = None
    quality_report: QualityReport = Field(default_factory=QualityReport)
    calibration: CalibrationDetails = Field(default_factory=CalibrationDetails)
    results: List[DiscMeasurement] = Field(default_factory=list)
    summary: AnalysisSummary = Field(default_factory=AnalysisSummary)
    warnings: List[str] = Field(default_factory=list)
    debug_artifacts: Dict[str, Any] = Field(default_factory=dict)


class AnalysisJobStatusResponse(BaseModel):
    analysis_id: str
    status: str = "queued"
    created_at: datetime
    updated_at: datetime
    image_filename: Optional[str] = None
    message: str = ""
    progress: float = 0.0
    current_stage: Optional[str] = None
    error: Optional[str] = None
    result_available: bool = False
    timings: Dict[str, float] = Field(default_factory=dict)
    status_url: Optional[str] = None
    result_url: Optional[str] = None


class AnalysisSubmissionResponse(BaseModel):
    analysis_id: str
    status: str = "queued"
    created_at: datetime
    message: str = ""
    status_url: str
    result_url: str


class AnalysisErrorRecord(BaseModel):
    analysis_id: str
    status: str = "failed"
    failed_at: datetime
    image_filename: Optional[str] = None
    current_stage: Optional[str] = None
    message: str = ""
    technical_details: Optional[str] = None
    timings: Dict[str, float] = Field(default_factory=dict)


class ReviewDiscUpdate(BaseModel):
    disc_id: str
    corrected_code: Optional[str] = None
    corrected_diameter_mm: Optional[float] = None
    operator_note: Optional[str] = None
    confirmed: bool = True


class SaveReviewRequest(BaseModel):
    operator_id: Optional[str] = None
    discs: List[ReviewDiscUpdate] = Field(default_factory=list)


class PersistedAnalysisRecord(BaseModel):
    analysis: AnalysisResponse
    storage: Dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime
    operator_id: Optional[str] = None
