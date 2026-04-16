from __future__ import annotations

from typing import Optional

from app.core.config import settings


def apply_no_zone_rule(diameter_mm: Optional[float]) -> float:
    if diameter_mm is None or diameter_mm < settings.DISC_DIAMETER_MM:
        return settings.DISC_DIAMETER_MM
    return round(float(diameter_mm), 2)


def calculate_inhibition_result(zone_diameter_px: float, disc_diameter_px: float, mm_per_pixel: float) -> float:
    if zone_diameter_px <= 0 or mm_per_pixel <= 0 or disc_diameter_px <= 0:
        return settings.DISC_DIAMETER_MM

    zone_mm = zone_diameter_px * mm_per_pixel
    disc_mm = disc_diameter_px * mm_per_pixel
    inhibition_halo_mm = zone_mm - disc_mm

    if inhibition_halo_mm <= 0.5:
        return settings.DISC_DIAMETER_MM

    return round(max(zone_mm, settings.DISC_DIAMETER_MM), 2)


def normalize_corrected_diameter(diameter_mm: Optional[float]) -> float:
    return apply_no_zone_rule(diameter_mm)
