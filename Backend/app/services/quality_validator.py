from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from app.models.schemas import PlateDetection, QualityReport


def validate_image_quality(image: np.ndarray, plate_detection: Optional[PlateDetection] = None) -> QualityReport:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    glare_fraction = float(np.mean(gray > 245))

    warnings = []
    review_required = False

    if blur_score < 35:
        warnings.append("Image appears blurry; recapture is recommended if measurements look wrong.")
        review_required = True
    elif blur_score < 60:
        warnings.append("Mild blur detected.")

    if brightness < 55:
        warnings.append("Image is dark; disc labels may need confirmation.")
        review_required = True
    elif brightness > 210:
        warnings.append("Image is very bright; glare may hide weak zones.")
        review_required = True

    if contrast < 18:
        warnings.append("Low contrast may reduce zone detection confidence.")
        review_required = True

    if glare_fraction > 0.04:
        warnings.append("Glare or reflection detected on the plate.")
        review_required = True

    clipped_plate = bool(plate_detection and plate_detection.clipped)
    severe_skew = bool(plate_detection and plate_detection.clip_fraction > 0.18)

    if clipped_plate:
        warnings.append("Plate is clipped near the image border.")
        review_required = True

    if severe_skew:
        warnings.append("Plate framing is off-center; verify edge-adjacent discs manually.")
        review_required = True

    return QualityReport(
        blur_score=round(blur_score, 2),
        brightness=round(brightness, 2),
        contrast=round(contrast, 2),
        glare_fraction=round(glare_fraction, 4),
        clipped_plate=clipped_plate,
        severe_skew=severe_skew,
        warnings=warnings,
        review_required=review_required,
    )
