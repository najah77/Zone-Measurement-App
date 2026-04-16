from __future__ import annotations

from statistics import median
from typing import Dict, List, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.utils.preprocessing import apply_clahe, preprocess_plate


def _sample_profile(gray: np.ndarray, cx: int, cy: int, angle_deg: float, max_radius: int) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    radii = np.arange(0, max_radius, dtype=float)
    xs = np.clip(cx + np.cos(angle_rad) * radii, 0, gray.shape[1] - 1).astype(int)
    ys = np.clip(cy + np.sin(angle_rad) * radii, 0, gray.shape[0] - 1).astype(int)
    return gray[ys, xs].astype(np.float32)


def _ring_occupancy(gray: np.ndarray, cx: int, cy: int, max_radius: int) -> np.ndarray:
    _, dark_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    yy, xx = np.indices(gray.shape)
    dists = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(int)
    occupancy = np.zeros(max_radius, dtype=np.float32)
    for radius in range(max_radius):
        ring = dists == radius
        ring_count = int(np.sum(ring))
        if ring_count == 0:
            continue
        occupancy[radius] = float(np.sum(dark_mask[ring] > 0)) / float(ring_count)
    return occupancy


def _boundary_from_profile(profile: np.ndarray, occupancy: np.ndarray, disc_radius: int) -> Tuple[float, float]:
    start = int(max(2, round(disc_radius * 1.05)))
    if len(profile) <= start + 8:
        return 0.0, 0.0

    smoothed = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 9), 0).flatten()
    signal = smoothed - np.mean(smoothed[int(len(smoothed) * 0.8) :])
    signal = signal[start:]
    occ = occupancy[start:]

    if len(signal) < 8:
        return 0.0, 0.0

    immediate_signal = float(np.mean(signal[: min(8, len(signal))]))
    peak_signal = float(np.max(signal[: max(8, len(signal) // 2)]))
    outer_occ = float(np.mean(occ[int(len(occ) * 0.7) :])) if len(occ) > 10 else float(np.mean(occ))
    immediate_occ = float(np.mean(occ[: min(8, len(occ))]))

    if immediate_signal < 6 and immediate_occ >= outer_occ * 0.8:
        return float(disc_radius * 2.0), 0.82

    if peak_signal < 5:
        return float(disc_radius * 2.0), 0.72

    threshold_signal = max(3.5, peak_signal * 0.35)
    occ_threshold = min(0.22, max(0.08, outer_occ * 0.6))

    for index in range(len(signal)):
        if index < 4:
            continue
        if signal[index] <= threshold_signal and occ[index] >= occ_threshold:
            boundary_radius = start + index
            stability = 1.0 - min(1.0, abs(immediate_occ - outer_occ))
            confidence = min(0.98, 0.55 + stability * 0.2 + min(0.2, peak_signal / 40.0))
            return float(boundary_radius * 2.0), float(confidence)

    return 0.0, 0.0


def measure_zone(image: np.ndarray, disc: Tuple[float, float, float]) -> Dict[str, float | bool | str]:
    preprocessed = preprocess_plate(image)
    gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY) if preprocessed.ndim == 3 else preprocessed
    gray = apply_clahe(gray, clip_limit=2.5)

    x, y, radius = disc
    cx, cy, disc_radius = int(round(x)), int(round(y)), int(round(radius))
    roi_radius = int(disc_radius * 8.0)

    x1 = max(0, cx - roi_radius)
    y1 = max(0, cy - roi_radius)
    x2 = min(gray.shape[1], cx + roi_radius)
    y2 = min(gray.shape[0], cy + roi_radius)

    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return {
            "diameter_px": float(disc_radius * 2.0),
            "confidence": 0.0,
            "review_required": True,
            "no_zone_fallback_used": True,
            "method": "radial-profile",
            "warning": "ROI is empty.",
        }

    roi_cx = cx - x1
    roi_cy = cy - y1
    max_radius = int(min(roi_cx, roi_cy, roi.shape[1] - roi_cx - 1, roi.shape[0] - roi_cy - 1))
    if max_radius <= disc_radius + 4:
        return {
            "diameter_px": float(disc_radius * 2.0),
            "confidence": 0.25,
            "review_required": True,
            "no_zone_fallback_used": True,
            "method": "radial-profile",
            "warning": "Disc is too close to the image border.",
        }

    occupancy = _ring_occupancy(roi, roi_cx, roi_cy, max_radius)
    boundaries: List[float] = []
    confidences: List[float] = []
    for angle in np.linspace(0, 330, 12):
        profile = _sample_profile(roi, roi_cx, roi_cy, float(angle), max_radius)
        diameter_px, confidence = _boundary_from_profile(profile, occupancy, disc_radius)
        if diameter_px > 0:
            boundaries.append(diameter_px)
            confidences.append(confidence)

    if not boundaries:
        return {
            "diameter_px": float(disc_radius * 2.0),
            "confidence": 0.22,
            "review_required": True,
            "no_zone_fallback_used": True,
            "method": "radial-profile",
            "warning": "No trustworthy zone boundary was found.",
        }

    measured_px = float(median(boundaries))
    spread = float(np.std(boundaries)) if len(boundaries) > 1 else 0.0
    confidence = float(np.mean(confidences))
    no_zone_fallback_used = measured_px <= (disc_radius * 2.0) + 2.0
    if no_zone_fallback_used:
        measured_px = float(disc_radius * 2.0)

    review_required = confidence < settings.MEASUREMENT_MIN_CONFIDENCE or spread > disc_radius * 0.9
    warning = ""
    if spread > disc_radius * 0.9:
        warning = "Zone edge varied across radial samples."
    elif confidence < settings.MEASUREMENT_MIN_CONFIDENCE:
        warning = "Zone confidence is low."
    elif no_zone_fallback_used:
        warning = "No visible inhibition zone detected; 6 mm disc fallback applied."

    return {
        "diameter_px": round(measured_px, 2),
        "confidence": round(max(0.0, min(0.99, confidence)), 3),
        "review_required": review_required,
        "no_zone_fallback_used": no_zone_fallback_used,
        "method": "radial-profile",
        "warning": warning,
    }


def detect_zones(image: np.ndarray, discs: List[Tuple[float, float, float]]) -> Tuple[List[float], List[float]]:
    diameters: List[float] = []
    confidences: List[float] = []
    for disc in discs:
        result = measure_zone(image, disc)
        diameters.append(float(result["diameter_px"]))
        confidences.append(float(result["confidence"]))
    return diameters, confidences
