from __future__ import annotations

from statistics import median
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from app.core.config import settings
from app.models.schemas import CalibrationDetails

STANDARD_DISC_DIAMETER_MM = settings.DISC_DIAMETER_MM


def _diameters_from_discs(discs: Sequence[Tuple[float, float, float]]) -> List[float]:
    return [float(radius) * 2.0 for _, _, radius in discs if radius > 0]


def build_calibration(discs: Sequence[Tuple[float, float, float]]) -> CalibrationDetails:
    diameters = _diameters_from_discs(discs)
    if not diameters:
        return CalibrationDetails(
            average_disc_diameter_px=0.0,
            mm_per_pixel=1.0,
            disc_count=0,
            spread_px=0.0,
            warnings=["No discs available for calibration; using 1.0 mm/px fallback."],
        )

    avg_disc_diameter_px = float(median(diameters))
    mm_per_pixel = STANDARD_DISC_DIAMETER_MM / avg_disc_diameter_px if avg_disc_diameter_px > 0 else 1.0
    spread_px = float(np.std(diameters)) if len(diameters) > 1 else 0.0
    warnings: List[str] = []
    if spread_px > avg_disc_diameter_px * 0.18:
        warnings.append("Detected disc sizes vary more than expected; calibration may need review.")

    return CalibrationDetails(
        average_disc_diameter_px=round(avg_disc_diameter_px, 3),
        mm_per_pixel=round(mm_per_pixel, 6),
        disc_count=len(diameters),
        spread_px=round(spread_px, 3),
        warnings=warnings,
    )


def calculate_mm_per_pixel(discs: Sequence[Tuple[float, float, float]]) -> float:
    return build_calibration(discs).mm_per_pixel


def pixels_to_mm(pixels: float, mm_per_pixel: float) -> float:
    return round(float(pixels) * float(mm_per_pixel), 2)


def mm_to_pixels(mm_value: float, mm_per_pixel: float) -> float:
    if mm_per_pixel <= 0:
        return 0.0
    return float(mm_value) / float(mm_per_pixel)
