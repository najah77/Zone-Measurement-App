from __future__ import annotations

from statistics import median
from typing import Dict, List, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.utils.preprocessing import apply_clahe, preprocess_plate


def _odd_kernel(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def _sample_profile(gray: np.ndarray, cx: int, cy: int, angle_deg: float, max_radius: int) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    radii = np.arange(0, max_radius, dtype=float)
    xs = np.clip(cx + np.cos(angle_rad) * radii, 0, gray.shape[1] - 1).astype(int)
    ys = np.clip(cy + np.sin(angle_rad) * radii, 0, gray.shape[0] - 1).astype(int)
    return gray[ys, xs].astype(np.float32)


def _extract_roi(gray: np.ndarray, cx: int, cy: int, disc_radius: int) -> Tuple[np.ndarray, int, int, int]:
    roi_radius = int(max(disc_radius * 8.0, disc_radius * 4.0 + 24))
    x1 = max(0, cx - roi_radius)
    y1 = max(0, cy - roi_radius)
    x2 = min(gray.shape[1], cx + roi_radius)
    y2 = min(gray.shape[0], cy + roi_radius)
    roi = gray[y1:y2, x1:x2]
    roi_cx = cx - x1
    roi_cy = cy - y1
    max_radius = int(min(roi_cx, roi_cy, roi.shape[1] - roi_cx - 1, roi.shape[0] - roi_cy - 1))
    return roi, roi_cx, roi_cy, max_radius


def _prepare_zone_maps(roi: np.ndarray, disc_radius: int) -> Dict[str, np.ndarray]:
    enhanced = apply_clahe(roi, clip_limit=3.2)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    block_size = _odd_kernel(disc_radius * 3, minimum=31)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        3,
    )

    background_kernel = _odd_kernel(disc_radius * 6, minimum=41)
    background = cv2.GaussianBlur(blurred, (background_kernel, background_kernel), 0)
    dark_diff = cv2.subtract(background, blurred)
    light_diff = cv2.subtract(blurred, background)
    contrast = cv2.max(dark_diff, light_diff)
    contrast = cv2.normalize(contrast, None, 0, 255, cv2.NORM_MINMAX)
    contrast = cv2.GaussianBlur(contrast, (5, 5), 0)

    adaptive_contrast = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        -2,
    )
    threshold_value = max(10, int(np.percentile(contrast, 72)))
    _, contrast_binary = cv2.threshold(contrast, threshold_value, 255, cv2.THRESH_BINARY)

    edge_low = max(18, int(np.percentile(contrast, 55) * 0.6))
    edge_high = max(edge_low + 18, int(np.percentile(contrast, 82) * 1.1))
    edges = cv2.Canny(blurred, 28, 92)
    contrast_edges = cv2.Canny(contrast, edge_low, edge_high)
    edge_map = cv2.bitwise_or(edges, contrast_edges)
    edge_map = cv2.dilate(edge_map, np.ones((3, 3), dtype=np.uint8), iterations=1)

    return {
        "enhanced": enhanced,
        "blurred": blurred,
        "adaptive": adaptive,
        "contrast": contrast,
        "adaptive_contrast": adaptive_contrast,
        "contrast_binary": contrast_binary,
        "edge_map": edge_map,
    }


def _ring_metric(image: np.ndarray, cx: int, cy: int, radii: np.ndarray) -> np.ndarray:
    yy, xx = np.indices(image.shape)
    dists = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
    metrics = np.zeros(len(radii), dtype=np.float32)
    for index, radius in enumerate(radii):
        ring = (dists >= radius - 0.5) & (dists < radius + 0.5)
        if np.any(ring):
            metrics[index] = float(np.mean(image[ring]))
    return metrics


def _gradient_strength(contrast: np.ndarray, edge_map: np.ndarray, cx: int, cy: int, disc_radius: int, max_radius: int) -> Tuple[float, float]:
    start = int(max(disc_radius * 1.1, 4))
    stop = int(max(start + 6, max_radius * 0.92))
    if stop <= start:
        return 0.0, 0.0
    radii = np.arange(start, stop, dtype=np.float32)
    contrast_profile = _ring_metric(contrast, cx, cy, radii)
    edge_profile = _ring_metric(edge_map, cx, cy, radii)
    return float(np.max(contrast_profile, initial=0.0)), float(np.max(edge_profile, initial=0.0) / 255.0)


def _annular_baseline_measurement(gray: np.ndarray, cx: int, cy: int, disc_radius: int, max_radius: int) -> Tuple[float, float]:
    start = int(max(disc_radius * 1.2, disc_radius + 12, 4))
    stop = int(max(start + 10, max_radius * 0.78))
    if stop <= start + 8:
        return 0.0, 0.0

    radii = np.arange(start, stop, dtype=np.float32)
    profile = _ring_metric(gray, cx, cy, radii)
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 11), 0).flatten()
    if len(profile) < 10:
        return 0.0, 0.0

    outer_slice = profile[int(len(profile) * 0.72) :]
    baseline = float(np.median(outer_slice)) if len(outer_slice) else float(np.median(profile))
    dark_deviation = baseline - profile
    light_deviation = profile - baseline
    deviation = dark_deviation if np.max(dark_deviation) >= np.max(light_deviation) else light_deviation
    peak_idx = int(np.argmax(deviation))
    peak = float(deviation[peak_idx])
    if peak < 16.0:
        return 0.0, 0.0

    release_threshold = max(1.5, peak * 0.26)
    boundary_idx = None
    for index in range(peak_idx + 2, len(deviation) - 4):
        window = deviation[index : index + 5]
        if float(np.mean(window)) <= release_threshold:
            boundary_idx = index
            break
    if boundary_idx is None:
        boundary_idx = min(len(deviation) - 1, peak_idx + max(6, int(len(deviation) * 0.12)))

    boundary_radius = float(radii[boundary_idx])
    if peak_idx <= 1 and boundary_radius > disc_radius * 2.2:
        return 0.0, 0.0
    contrast_drop = max(0.0, peak - float(np.mean(deviation[max(0, boundary_idx - 3) : boundary_idx + 1])))
    confidence = min(0.93, 0.48 + min(0.24, peak / 24.0) + min(0.12, contrast_drop / 16.0))
    return boundary_radius * 2.0, confidence


def _boundary_from_profile(intensity: np.ndarray, contrast: np.ndarray, edges: np.ndarray, disc_radius: int) -> Tuple[float, float]:
    start = int(max(2, round(disc_radius * 1.08)))
    limit = int(len(intensity) * 0.9)
    if limit <= start + 6:
        return 0.0, 0.0

    smoothed_intensity = cv2.GaussianBlur(intensity.reshape(-1, 1), (1, 9), 0).flatten()
    smoothed_contrast = cv2.GaussianBlur(contrast.reshape(-1, 1), (1, 7), 0).flatten()
    smoothed_edges = cv2.GaussianBlur(edges.reshape(-1, 1), (1, 5), 0).flatten()
    gradient = np.abs(np.gradient(smoothed_intensity))
    response = smoothed_contrast + (gradient * 1.6) + (smoothed_edges / 255.0 * 24.0)
    response = response[start:limit]
    if len(response) < 6:
        return 0.0, 0.0

    baseline = float(np.median(response[int(len(response) * 0.72) :])) if len(response) > 10 else float(np.median(response))
    peak = float(np.max(response))
    if peak - baseline < 2.3:
        return 0.0, 0.0

    threshold = baseline + max(1.8, (peak - baseline) * 0.28)
    strong = np.where(response >= threshold)[0]
    if strong.size == 0:
        return 0.0, 0.0

    boundary_radius = start + int(strong[-1])
    density = float(strong.size) / float(len(response))
    confidence = min(0.96, 0.42 + min(0.28, (peak - baseline) / 22.0) + min(0.12, density * 0.2))
    return float(boundary_radius * 2.0), float(confidence)


def _radial_measurement(
    maps: Dict[str, np.ndarray],
    cx: int,
    cy: int,
    disc_radius: int,
    max_radius: int,
) -> Tuple[float, float, float]:
    diameters: List[float] = []
    confidences: List[float] = []
    for angle in np.linspace(0, 337.5, 16):
        intensity = _sample_profile(maps["blurred"], cx, cy, float(angle), max_radius)
        contrast = _sample_profile(maps["contrast"], cx, cy, float(angle), max_radius)
        edges = _sample_profile(maps["edge_map"], cx, cy, float(angle), max_radius)
        diameter_px, confidence = _boundary_from_profile(intensity, contrast, edges, disc_radius)
        if diameter_px > 0:
            diameters.append(diameter_px)
            confidences.append(confidence)

    if not diameters:
        return 0.0, 0.0, 0.0
    return float(median(diameters)), float(np.mean(confidences)), float(np.std(diameters))


def _build_zone_mask(maps: Dict[str, np.ndarray], cx: int, cy: int, disc_radius: int) -> np.ndarray:
    zone_mask = cv2.bitwise_or(maps["adaptive"], maps["adaptive_contrast"])
    zone_mask = cv2.bitwise_or(zone_mask, maps["contrast_binary"])
    cv2.circle(zone_mask, (cx, cy), int(round(disc_radius * 0.92)), 255, -1)

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (_odd_kernel(disc_radius * 0.7, minimum=7), _odd_kernel(disc_radius * 0.7, minimum=7)),
    )
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    zone_mask = cv2.morphologyEx(zone_mask, cv2.MORPH_CLOSE, close_kernel)
    zone_mask = cv2.morphologyEx(zone_mask, cv2.MORPH_OPEN, open_kernel)
    return zone_mask


def _contour_candidates(
    zone_mask: np.ndarray,
    edge_map: np.ndarray,
    cx: int,
    cy: int,
    disc_radius: int,
    max_radius: int,
) -> List[Tuple[float, float, str]]:
    candidates: List[Tuple[float, float, str]] = []
    contours, _ = cv2.findContours(zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < np.pi * (disc_radius * 1.08) ** 2:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        contains_center = cv2.pointPolygonTest(contour, (float(cx), float(cy)), False) >= 0
        (circle_x, circle_y), circle_radius = cv2.minEnclosingCircle(contour)
        center_distance = float(np.hypot(circle_x - cx, circle_y - cy))
        if circle_radius <= disc_radius * 1.05 or circle_radius >= max_radius * 0.97:
            continue
        if not contains_center and center_distance > disc_radius * 0.9:
            continue

        equivalent_radius = float(np.sqrt(area / np.pi))
        circularity = float((4.0 * np.pi * area) / (perimeter * perimeter))
        irregularity = abs(circle_radius - equivalent_radius) / max(circle_radius, 1.0)
        diameter = float(np.median([circle_radius * 2.0, equivalent_radius * 2.0]))
        confidence = 0.46 + min(0.16, max(0.0, circularity) * 0.18) + min(0.14, max(0.0, 1.0 - irregularity) * 0.14)
        if contains_center:
            confidence += 0.08
        if center_distance > disc_radius * 0.45:
            confidence -= 0.08
        candidates.append((diameter, min(0.94, confidence), "contour"))

    edge_contours, _ = cv2.findContours(edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in edge_contours:
        if len(contour) < 10:
            continue
        (circle_x, circle_y), circle_radius = cv2.minEnclosingCircle(contour)
        center_distance = float(np.hypot(circle_x - cx, circle_y - cy))
        if circle_radius <= disc_radius * 1.08 or circle_radius >= max_radius * 0.97:
            continue
        if center_distance > disc_radius * 0.9:
            continue
        candidates.append((float(circle_radius * 2.0), 0.42, "contour-circle-fit"))

    return candidates


def _estimate_from_energy(contrast: np.ndarray, edge_map: np.ndarray, cx: int, cy: int, disc_radius: int, max_radius: int) -> Tuple[float, float]:
    start = int(max(disc_radius * 1.08, 4))
    stop = int(max(start + 6, max_radius * 0.92))
    if stop <= start:
        return 0.0, 0.0

    radii = np.arange(start, stop, dtype=np.float32)
    contrast_profile = _ring_metric(contrast, cx, cy, radii)
    edge_profile = _ring_metric(edge_map, cx, cy, radii) / 255.0
    signal = cv2.GaussianBlur((contrast_profile + edge_profile * 18.0).reshape(-1, 1), (1, 7), 0).flatten()
    baseline = float(np.median(signal[int(len(signal) * 0.72) :])) if len(signal) > 10 else float(np.median(signal))
    peak = float(np.max(signal))
    if peak - baseline < 1.6:
        return 0.0, 0.0

    threshold = baseline + max(1.1, (peak - baseline) * 0.18)
    strong = np.where(signal >= threshold)[0]
    if strong.size == 0:
        return 0.0, 0.0
    boundary_radius = start + int(strong[-1])
    confidence = min(0.62, 0.28 + min(0.22, (peak - baseline) / 18.0))
    return float(boundary_radius * 2.0), float(confidence)


def measure_zone(image: np.ndarray, disc: Tuple[float, float, float]) -> Dict[str, float | bool | str]:
    preprocessed = preprocess_plate(image)
    gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY) if preprocessed.ndim == 3 else preprocessed
    gray = apply_clahe(gray, clip_limit=2.8)

    x, y, radius = disc
    cx, cy, disc_radius = int(round(x)), int(round(y)), int(round(radius))
    roi, roi_cx, roi_cy, max_radius = _extract_roi(gray, cx, cy, disc_radius)
    disc_diameter = float(disc_radius * 2.0)

    if roi.size == 0:
        return {
            "diameter_px": disc_diameter,
            "confidence": 0.0,
            "review_required": True,
            "no_zone_fallback_used": True,
            "method": "hybrid-zone-detector",
            "warning": "ROI is empty.",
        }

    if max_radius <= disc_radius + 4:
        return {
            "diameter_px": disc_diameter,
            "confidence": 0.25,
            "review_required": True,
            "no_zone_fallback_used": True,
            "method": "hybrid-zone-detector",
            "warning": "Disc is too close to the image border.",
        }

    maps = _prepare_zone_maps(roi, disc_radius)
    gradient_strength, edge_density = _gradient_strength(maps["contrast"], maps["edge_map"], roi_cx, roi_cy, disc_radius, max_radius)
    annular_diameter, annular_confidence = _annular_baseline_measurement(maps["blurred"], roi_cx, roi_cy, disc_radius, max_radius)
    radial_diameter, radial_confidence, radial_spread = _radial_measurement(maps, roi_cx, roi_cy, disc_radius, max_radius)
    zone_mask = _build_zone_mask(maps, roi_cx, roi_cy, disc_radius)
    contour_candidates = _contour_candidates(zone_mask, maps["edge_map"], roi_cx, roi_cy, disc_radius, max_radius)
    contour_consistent_with_radial = any(
        abs(candidate[0] - radial_diameter) <= max(20.0, radial_diameter * 0.45) for candidate in contour_candidates
    ) if radial_diameter > 0 else False
    if annular_diameter == 0.0 and not contour_consistent_with_radial:
        contour_candidates = []

    candidates: List[Tuple[float, float, str]] = []
    if annular_diameter > disc_diameter:
        candidates.append((annular_diameter, annular_confidence, "annular-baseline"))
    if radial_diameter > disc_diameter and (annular_diameter > 0.0 or contour_consistent_with_radial):
        candidates.append((radial_diameter, radial_confidence, "radial-profile"))
    candidates.extend([candidate for candidate in contour_candidates if candidate[0] > disc_diameter * 1.02])

    if not candidates:
        if annular_diameter == 0.0 and not contour_candidates:
            return {
                "diameter_px": disc_diameter,
                "confidence": 0.78,
                "review_required": False,
                "no_zone_fallback_used": True,
                "method": "annular-baseline",
                "warning": "No visible inhibition zone gradient was found.",
            }

        estimated_diameter, estimated_confidence = _estimate_from_energy(
            maps["contrast"],
            maps["edge_map"],
            roi_cx,
            roi_cy,
            disc_radius,
            max_radius,
        )
        has_gradient = gradient_strength >= 12.0 or edge_density >= 0.035
        if estimated_diameter > disc_diameter * 1.02 and has_gradient:
            warning = "Zone boundary is weak; estimated from low-contrast energy."
            return {
                "diameter_px": round(estimated_diameter, 2),
                "confidence": round(max(0.18, min(0.62, estimated_confidence)), 3),
                "review_required": True,
                "no_zone_fallback_used": False,
                "method": "energy-fallback",
                "warning": warning,
            }

        if not has_gradient:
            return {
                "diameter_px": disc_diameter,
                "confidence": 0.78,
                "review_required": False,
                "no_zone_fallback_used": True,
                "method": "hybrid-zone-detector",
                "warning": "No visible inhibition zone gradient was found.",
            }

        return {
            "diameter_px": round(max(disc_diameter, estimated_diameter or disc_diameter), 2),
            "confidence": round(max(0.16, estimated_confidence), 3),
            "review_required": True,
            "no_zone_fallback_used": False,
            "method": "energy-fallback",
            "warning": "Zone is uncertain; estimated boundary should be reviewed.",
        }

    candidates = sorted(candidates, key=lambda item: (item[1], item[0]), reverse=True)
    best_diameter, best_confidence, best_method = candidates[0]
    close_diameters = [
        diameter
        for diameter, confidence, _ in candidates
        if abs(diameter - best_diameter) <= max(8.0, disc_diameter * 0.22) and confidence >= best_confidence - 0.16
    ]
    if close_diameters:
        measured_px = float(median(close_diameters))
    else:
        measured_px = float(best_diameter)

    no_zone_fallback_used = measured_px <= disc_diameter * 1.02 and gradient_strength < 12.0 and edge_density < 0.035
    if no_zone_fallback_used:
        measured_px = disc_diameter

    review_required = (
        best_confidence < settings.MEASUREMENT_MIN_CONFIDENCE
        or radial_spread > disc_radius * 1.2
        or best_method not in {"radial-profile", "annular-baseline"}
    )
    warning = ""
    if no_zone_fallback_used:
        warning = "No visible inhibition zone gradient was found."
    elif radial_spread > disc_radius * 1.2:
        warning = "Zone edge is irregular across radial samples."
    elif best_confidence < settings.MEASUREMENT_MIN_CONFIDENCE:
        warning = "Zone confidence is low; confirm the boundary."
    elif best_method not in {"radial-profile", "annular-baseline"}:
        warning = "Zone boundary was recovered from contour fallback."

    return {
        "diameter_px": round(max(measured_px, disc_diameter), 2),
        "confidence": round(max(0.0, min(0.97, best_confidence)), 3),
        "review_required": review_required,
        "no_zone_fallback_used": no_zone_fallback_used,
        "method": best_method,
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
