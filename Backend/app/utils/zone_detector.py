from __future__ import annotations

from statistics import median
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.models.schemas import PlateDetection
from app.utils.plate_extractor import build_plate_mask
from app.utils.preprocessing import apply_clahe, preprocess_plate


def _odd_kernel(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def _sample_profile(
    gray: np.ndarray,
    cx: int,
    cy: int,
    angle_deg: float,
    max_radius: int,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    radii = np.arange(0, max_radius, dtype=float)
    xs = np.clip(cx + np.cos(angle_rad) * radii, 0, gray.shape[1] - 1).astype(int)
    ys = np.clip(cy + np.sin(angle_rad) * radii, 0, gray.shape[0] - 1).astype(int)
    values = gray[ys, xs].astype(np.float32)
    if valid_mask is None:
        return values
    valid = valid_mask[ys, xs] > 0
    if not np.any(valid):
        return np.array([], dtype=np.float32)
    last_valid_index = int(np.max(np.where(valid)[0])) + 1
    return values[:last_valid_index]


def _extract_roi(gray: np.ndarray, cx: int, cy: int, disc_radius: int) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    base_roi_radius = int(max(disc_radius * 8.0, disc_radius * 4.0 + 24))
    nearest_border = min(cx, cy, gray.shape[1] - cx - 1, gray.shape[0] - cy - 1)
    if nearest_border >= base_roi_radius + 2:
        x1 = max(0, cx - base_roi_radius)
        y1 = max(0, cy - base_roi_radius)
        x2 = min(gray.shape[1], cx + base_roi_radius)
        y2 = min(gray.shape[0], cy + base_roi_radius)
        roi = gray[y1:y2, x1:x2]
        roi_valid = np.full(roi.shape, 255, dtype=np.uint8)
        roi_cx = cx - x1
        roi_cy = cy - y1
        max_radius = int(min(roi_cx, roi_cy, roi.shape[1] - roi_cx - 1, roi.shape[0] - roi_cy - 1))
        return roi, roi_valid, roi_cx, roi_cy, max_radius

    roi_radius = int(max(disc_radius * 9.0, disc_radius * 4.75 + 32))
    pad = roi_radius + 4
    fill_value = int(np.median(gray))
    padded = cv2.copyMakeBorder(
        gray,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=fill_value,
    )
    valid_mask = np.zeros((gray.shape[0] + pad * 2, gray.shape[1] + pad * 2), dtype=np.uint8)
    valid_mask[pad : pad + gray.shape[0], pad : pad + gray.shape[1]] = 255

    padded_cx = cx + pad
    padded_cy = cy + pad
    x1 = int(round(padded_cx - roi_radius))
    y1 = int(round(padded_cy - roi_radius))
    x2 = int(round(padded_cx + roi_radius))
    y2 = int(round(padded_cy + roi_radius))

    roi = padded[y1:y2, x1:x2]
    roi_valid = valid_mask[y1:y2, x1:x2]
    roi_cx = int(round(padded_cx - x1))
    roi_cy = int(round(padded_cy - y1))
    max_radius = max(0, int(min(roi_radius - 2, min(roi.shape[:2]) / 2 - 2)))
    return roi, roi_valid, roi_cx, roi_cy, max_radius


def _build_plate_valid_mask(
    image: np.ndarray,
    plate_detection: PlateDetection | None = None,
) -> np.ndarray | None:
    if image is None or image.size == 0:
        return None

    height, width = image.shape[:2]
    if plate_detection and plate_detection.shape == "rectangle":
        inset_ratio = settings.ZONE_PLATE_MASK_INSET_RATIO_RECTANGLE
        inset_x = max(6, int(round(width * max(0.0, (1.0 - inset_ratio)) * 0.5)))
        inset_y = max(6, int(round(height * max(0.0, (1.0 - inset_ratio)) * 0.5)))
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(
            mask,
            (inset_x, inset_y),
            (max(inset_x + 1, width - inset_x), max(inset_y + 1, height - inset_y)),
            255,
            -1,
        )
        return mask

    if plate_detection and plate_detection.shape == "circle":
        return build_plate_mask(
            image,
            inset_ratio=settings.ZONE_PLATE_MASK_INSET_RATIO_CIRCLE,
        )

    return None


def build_disc_assignment_masks(
    image_shape: tuple[int, int] | tuple[int, int, int],
    discs: Sequence[Tuple[float, float, float]],
    *,
    plate_detection: PlateDetection | None = None,
) -> List[np.ndarray]:
    if not discs:
        return []

    height, width = image_shape[:2]
    global_plate_mask = _build_plate_valid_mask(np.zeros((height, width), dtype=np.uint8), plate_detection=plate_detection)
    if global_plate_mask is None:
        global_plate_mask = np.full((height, width), 255, dtype=np.uint8)

    centers = np.array([(disc[0], disc[1]) for disc in discs], dtype=np.float32)
    yy, xx = np.indices((height, width), dtype=np.float32)
    distances = (xx[None, :, :] - centers[:, 0, None, None]) ** 2 + (yy[None, :, :] - centers[:, 1, None, None]) ** 2
    assignments = np.argmin(distances, axis=0)

    cell_masks: List[np.ndarray] = []
    for index, (cx, cy, radius) in enumerate(discs):
        cell_mask = np.zeros((height, width), dtype=np.uint8)
        cell_mask[assignments == index] = 255
        cell_mask = cv2.bitwise_and(cell_mask, global_plate_mask)
        cv2.circle(
            cell_mask,
            (int(round(cx)), int(round(cy))),
            max(6, int(round(radius * 1.12))),
            255,
            -1,
        )
        cell_mask = cv2.morphologyEx(
            cell_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        )
        cell_masks.append(cell_mask)
    return cell_masks


def _crop_global_mask_to_roi(
    global_mask: np.ndarray | None,
    roi_shape: tuple[int, int],
    x1: int,
    y1: int,
) -> np.ndarray | None:
    if global_mask is None or global_mask.size == 0:
        return None

    roi_height, roi_width = roi_shape
    aligned = np.zeros((roi_height, roi_width), dtype=np.uint8)

    source_x1 = max(0, x1)
    source_y1 = max(0, y1)
    source_x2 = min(global_mask.shape[1], x1 + roi_width)
    source_y2 = min(global_mask.shape[0], y1 + roi_height)
    if source_x2 <= source_x1 or source_y2 <= source_y1:
        return aligned

    dest_x1 = max(0, -x1)
    dest_y1 = max(0, -y1)
    dest_x2 = dest_x1 + (source_x2 - source_x1)
    dest_y2 = dest_y1 + (source_y2 - source_y1)
    aligned[dest_y1:dest_y2, dest_x1:dest_x2] = global_mask[source_y1:source_y2, source_x1:source_x2]
    return aligned


def _prepare_zone_maps(roi: np.ndarray, disc_radius: int, valid_mask: np.ndarray | None = None) -> Dict[str, np.ndarray]:
    if valid_mask is not None and np.any(valid_mask > 0):
        fill_value = int(np.median(roi[valid_mask > 0]))
        roi = roi.copy()
        roi[valid_mask == 0] = fill_value

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

    if valid_mask is not None:
        enhanced[valid_mask == 0] = 0
        blurred[valid_mask == 0] = 0
        adaptive[valid_mask == 0] = 0
        contrast[valid_mask == 0] = 0
        adaptive_contrast[valid_mask == 0] = 0
        contrast_binary[valid_mask == 0] = 0
        edge_map[valid_mask == 0] = 0

    return {
        "enhanced": enhanced,
        "blurred": blurred,
        "adaptive": adaptive,
        "contrast": contrast,
        "adaptive_contrast": adaptive_contrast,
        "contrast_binary": contrast_binary,
        "edge_map": edge_map,
    }


def _ring_metric(image: np.ndarray, cx: int, cy: int, radii: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    yy, xx = np.indices(image.shape)
    dists = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
    metrics = np.zeros(len(radii), dtype=np.float32)
    for index, radius in enumerate(radii):
        ring = (dists >= radius - 0.5) & (dists < radius + 0.5)
        if valid_mask is not None:
            ring &= valid_mask > 0
        if np.any(ring):
            metrics[index] = float(np.mean(image[ring]))
    return metrics


def _gradient_strength(
    contrast: np.ndarray,
    edge_map: np.ndarray,
    cx: int,
    cy: int,
    disc_radius: int,
    max_radius: int,
    valid_mask: np.ndarray | None = None,
) -> Tuple[float, float]:
    start = int(max(disc_radius * 1.1, 4))
    stop = int(max(start + 6, max_radius * 0.92))
    if stop <= start:
        return 0.0, 0.0
    radii = np.arange(start, stop, dtype=np.float32)
    contrast_profile = _ring_metric(contrast, cx, cy, radii, valid_mask=valid_mask)
    edge_profile = _ring_metric(edge_map, cx, cy, radii, valid_mask=valid_mask)
    return float(np.max(contrast_profile, initial=0.0)), float(np.max(edge_profile, initial=0.0) / 255.0)


def _annular_baseline_measurement(
    gray: np.ndarray,
    cx: int,
    cy: int,
    disc_radius: int,
    max_radius: int,
    valid_mask: np.ndarray | None = None,
) -> Tuple[float, float]:
    start = int(max(disc_radius * 1.2, disc_radius + 12, 4))
    stop = int(max(start + 10, max_radius * 0.78))
    if stop <= start + 8:
        return 0.0, 0.0

    radii = np.arange(start, stop, dtype=np.float32)
    profile = _ring_metric(gray, cx, cy, radii, valid_mask=valid_mask)
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

    peak_idx = int(np.argmax(response))
    release_threshold = baseline + max(1.4, (peak - baseline) * 0.18)
    boundary_idx = None
    for index in range(peak_idx + 2, len(response) - 3):
        window = response[index : index + 4]
        if float(np.mean(window)) <= release_threshold:
            boundary_idx = index
            break
    if boundary_idx is None:
        boundary_idx = min(len(response) - 1, peak_idx + max(3, int(len(response) * 0.08)))

    if boundary_idx <= peak_idx and peak_idx > 0:
        boundary_idx = min(len(response) - 1, peak_idx + 1)

    strong = np.where(response >= (baseline + max(1.8, (peak - baseline) * 0.28)))[0]
    if strong.size == 0 and boundary_idx <= 0:
        return 0.0, 0.0

    boundary_radius = start + int(boundary_idx)
    density = float(strong.size) / float(len(response))
    confidence = min(0.96, 0.42 + min(0.28, (peak - baseline) / 22.0) + min(0.12, density * 0.2))
    return float(boundary_radius * 2.0), float(confidence)


def _radial_measurement(
    maps: Dict[str, np.ndarray],
    cx: int,
    cy: int,
    disc_radius: int,
    max_radius: int,
) -> Tuple[float, float, float, float]:
    diameters: List[float] = []
    confidences: List[float] = []
    coverages: List[float] = []
    for angle in np.linspace(0, 337.5, 16):
        intensity = _sample_profile(maps["blurred"], cx, cy, float(angle), max_radius, valid_mask=maps.get("valid_mask"))
        contrast = _sample_profile(maps["contrast"], cx, cy, float(angle), max_radius, valid_mask=maps.get("valid_mask"))
        edges = _sample_profile(maps["edge_map"], cx, cy, float(angle), max_radius, valid_mask=maps.get("valid_mask"))
        if intensity.size == 0 or contrast.size == 0 or edges.size == 0:
            continue
        diameter_px, confidence = _boundary_from_profile(intensity, contrast, edges, disc_radius)
        if diameter_px > 0:
            diameters.append(diameter_px)
            confidences.append(confidence)
            coverages.append(min(1.0, float(len(intensity)) / float(max(max_radius, 1))))

    if not diameters:
        return 0.0, 0.0, 0.0, 0.0

    clipped_fraction = float(np.mean([coverage < 0.92 for coverage in coverages])) if coverages else 0.0
    if clipped_fraction >= 0.25 and len(diameters) >= 4:
        fully_observed = [
            diameter
            for diameter, coverage in zip(diameters, coverages)
            if coverage >= 0.95
        ]
        if len(fully_observed) >= 4:
            dominant_threshold = float(np.percentile(fully_observed, 70))
            dominant_cluster = [diameter for diameter in fully_observed if diameter >= dominant_threshold]
            chosen_diameter = float(np.mean(dominant_cluster))
        elif len(fully_observed) >= 3:
            chosen_diameter = float(np.mean(sorted(fully_observed)[-2:]))
        elif fully_observed:
            chosen_diameter = float(np.median(fully_observed))
        else:
            chosen_diameter = float(np.percentile(diameters, 58))
    else:
        chosen_diameter = float(median(diameters))
    confidence = float(np.mean(confidences))
    if clipped_fraction >= 0.25:
        confidence = max(0.0, confidence - min(0.08, clipped_fraction * 0.12))
    return chosen_diameter, confidence, float(np.std(diameters)), clipped_fraction


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
    valid_mask = maps.get("valid_mask")
    if valid_mask is not None:
        zone_mask[valid_mask == 0] = 0
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


def _estimate_from_energy(
    contrast: np.ndarray,
    edge_map: np.ndarray,
    cx: int,
    cy: int,
    disc_radius: int,
    max_radius: int,
    valid_mask: np.ndarray | None = None,
) -> Tuple[float, float]:
    start = int(max(disc_radius * 1.08, 4))
    stop = int(max(start + 6, max_radius * 0.92))
    if stop <= start:
        return 0.0, 0.0

    radii = np.arange(start, stop, dtype=np.float32)
    contrast_profile = _ring_metric(contrast, cx, cy, radii, valid_mask=valid_mask)
    edge_profile = _ring_metric(edge_map, cx, cy, radii, valid_mask=valid_mask) / 255.0
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


def _build_zone_overlay(
    roi: np.ndarray,
    cx: int,
    cy: int,
    disc_radius: int,
    diameter_px: float,
) -> np.ndarray:
    overlay = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR) if roi.ndim == 2 else roi.copy()
    cv2.circle(overlay, (cx, cy), int(round(disc_radius)), (0, 255, 0), 2)
    if diameter_px > 0:
        cv2.circle(overlay, (cx, cy), int(round(diameter_px / 2.0)), (255, 196, 0), 2)
    return overlay


def _measure_zone_internal(
    image: np.ndarray,
    disc: Tuple[float, float, float],
    *,
    plate_detection: PlateDetection | None = None,
    extra_valid_mask: np.ndarray | None = None,
    return_debug: bool = False,
) -> tuple[Dict[str, float | bool | str], Dict[str, object]]:
    preprocessed = preprocess_plate(image)
    gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY) if preprocessed.ndim == 3 else preprocessed
    gray = apply_clahe(gray, clip_limit=2.8)

    x, y, radius = disc
    cx, cy, disc_radius = int(round(x)), int(round(y)), int(round(radius))
    roi, roi_valid_mask, roi_cx, roi_cy, max_radius = _extract_roi(gray, cx, cy, disc_radius)
    plate_valid_mask = _build_plate_valid_mask(image, plate_detection=plate_detection)
    roi_x1 = cx - roi_cx
    roi_y1 = cy - roi_cy
    roi_plate_mask = _crop_global_mask_to_roi(plate_valid_mask, roi.shape, roi_x1, roi_y1)
    if roi_plate_mask is not None:
        roi_valid_mask = cv2.bitwise_and(roi_valid_mask, roi_plate_mask)
    roi_extra_mask = _crop_global_mask_to_roi(extra_valid_mask, roi.shape, roi_x1, roi_y1)
    if roi_extra_mask is not None:
        roi_valid_mask = cv2.bitwise_and(roi_valid_mask, roi_extra_mask)
    disc_diameter = float(disc_radius * 2.0)
    debug: Dict[str, object] = {
        "roi": roi,
        "valid_mask": roi_valid_mask,
        "plate_mask": roi_plate_mask,
        "extra_valid_mask": roi_extra_mask,
        "preprocessed_gray": gray,
        "disc_center_roi": (roi_cx, roi_cy),
        "disc_radius_px": disc_radius,
        "max_radius_px": max_radius,
    }

    if roi.size == 0:
        result = {
            "diameter_px": disc_diameter,
            "confidence": 0.0,
            "review_required": True,
            "no_zone_fallback_used": True,
            "method": "hybrid-zone-detector",
            "warning": "ROI is empty.",
        }
        return result, debug

    if max_radius <= disc_radius + 4:
        result = {
            "diameter_px": disc_diameter,
            "confidence": 0.25,
            "review_required": True,
            "no_zone_fallback_used": True,
            "method": "hybrid-zone-detector",
            "warning": "Disc is too close to the image border.",
        }
        debug["overlay"] = _build_zone_overlay(roi, roi_cx, roi_cy, disc_radius, disc_diameter)
        return result, debug

    maps = _prepare_zone_maps(roi, disc_radius, valid_mask=roi_valid_mask)
    maps["valid_mask"] = roi_valid_mask
    gradient_strength, edge_density = _gradient_strength(
        maps["contrast"],
        maps["edge_map"],
        roi_cx,
        roi_cy,
        disc_radius,
        max_radius,
        valid_mask=roi_valid_mask,
    )
    annular_diameter, annular_confidence = _annular_baseline_measurement(
        maps["blurred"],
        roi_cx,
        roi_cy,
        disc_radius,
        max_radius,
        valid_mask=roi_valid_mask,
    )
    radial_diameter, radial_confidence, radial_spread, clipped_fraction = _radial_measurement(
        maps,
        roi_cx,
        roi_cy,
        disc_radius,
        max_radius,
    )
    zone_mask = _build_zone_mask(maps, roi_cx, roi_cy, disc_radius)
    contour_candidates = _contour_candidates(zone_mask, maps["edge_map"], roi_cx, roi_cy, disc_radius, max_radius)
    debug.update(
        {
            "enhanced": maps["enhanced"],
            "contrast": maps["contrast"],
            "edge_map": maps["edge_map"],
            "zone_mask": zone_mask,
            "gradient_strength": round(float(gradient_strength), 3),
            "edge_density": round(float(edge_density), 4),
            "annular_diameter_px": round(float(annular_diameter), 2),
            "annular_confidence": round(float(annular_confidence), 3),
            "radial_diameter_px": round(float(radial_diameter), 2),
            "radial_confidence": round(float(radial_confidence), 3),
            "radial_spread_px": round(float(radial_spread), 3),
            "radial_clipped_fraction": round(float(clipped_fraction), 3),
            "contour_candidates": [
                {
                    "diameter_px": round(float(diameter), 2),
                    "confidence": round(float(confidence), 3),
                    "method": method,
                }
                for diameter, confidence, method in contour_candidates
            ],
        }
    )
    contour_consistent_with_radial = any(
        abs(candidate[0] - radial_diameter) <= max(20.0, radial_diameter * 0.45) for candidate in contour_candidates
    ) if radial_diameter > 0 else False
    if annular_diameter == 0.0 and not contour_consistent_with_radial:
        contour_candidates = []

    candidates: List[Tuple[float, float, str]] = []
    if annular_diameter > disc_diameter:
        candidates.append((annular_diameter, annular_confidence, "annular-baseline"))
    radial_has_strong_evidence = radial_confidence >= 0.72 or gradient_strength >= 20.0 or edge_density >= 0.06
    if radial_diameter > disc_diameter and (annular_diameter > 0.0 or contour_consistent_with_radial or radial_has_strong_evidence):
        candidates.append((radial_diameter, radial_confidence, "radial-profile"))
        if (
            annular_diameter > 0.0
            and radial_diameter >= annular_diameter * 1.28
            and radial_confidence >= max(0.68, annular_confidence - 0.12)
            and gradient_strength >= 20.0
            and (clipped_fraction >= 0.2 or contour_consistent_with_radial or annular_confidence < 0.72)
        ):
            candidates.append((radial_diameter, min(0.9, radial_confidence + 0.09), "radial-large-zone"))
    candidates.extend([candidate for candidate in contour_candidates if candidate[0] > disc_diameter * 1.02])

    if not candidates:
        if annular_diameter == 0.0 and not contour_candidates:
            result = {
                "diameter_px": disc_diameter,
                "confidence": 0.78,
                "review_required": False,
                "no_zone_fallback_used": True,
                "method": "annular-baseline",
                "warning": "No visible inhibition zone gradient was found.",
            }
            debug["overlay"] = _build_zone_overlay(roi, roi_cx, roi_cy, disc_radius, disc_diameter)
            return result, debug

        estimated_diameter, estimated_confidence = _estimate_from_energy(
            maps["contrast"],
            maps["edge_map"],
            roi_cx,
            roi_cy,
            disc_radius,
            max_radius,
            valid_mask=roi_valid_mask,
        )
        has_gradient = gradient_strength >= 12.0 or edge_density >= 0.035
        weak_energy_only = estimated_diameter <= disc_diameter * 1.18 and gradient_strength < 16.0 and edge_density < 0.05
        if weak_energy_only:
            has_gradient = False
        if estimated_diameter > disc_diameter * 1.02 and has_gradient:
            warning = "Zone boundary is weak; estimated from low-contrast energy."
            result = {
                "diameter_px": round(estimated_diameter, 2),
                "confidence": round(max(0.18, min(0.62, estimated_confidence)), 3),
                "review_required": True,
                "no_zone_fallback_used": False,
                "method": "energy-fallback",
                "warning": warning,
            }
            debug["overlay"] = _build_zone_overlay(roi, roi_cx, roi_cy, disc_radius, estimated_diameter)
            debug["estimated_energy_diameter_px"] = round(float(estimated_diameter), 2)
            debug["estimated_energy_confidence"] = round(float(estimated_confidence), 3)
            return result, debug

        if not has_gradient:
            result = {
                "diameter_px": disc_diameter,
                "confidence": 0.78,
                "review_required": False,
                "no_zone_fallback_used": True,
                "method": "hybrid-zone-detector",
                "warning": "No visible inhibition zone gradient was found.",
            }
            debug["overlay"] = _build_zone_overlay(roi, roi_cx, roi_cy, disc_radius, disc_diameter)
            return result, debug

        result = {
            "diameter_px": round(max(disc_diameter, estimated_diameter or disc_diameter), 2),
            "confidence": round(max(0.16, estimated_confidence), 3),
            "review_required": True,
            "no_zone_fallback_used": False,
            "method": "energy-fallback",
            "warning": "Zone is uncertain; estimated boundary should be reviewed.",
        }
        debug["overlay"] = _build_zone_overlay(
            roi,
            roi_cx,
            roi_cy,
            disc_radius,
            max(disc_diameter, estimated_diameter or disc_diameter),
        )
        debug["estimated_energy_diameter_px"] = round(float(estimated_diameter), 2)
        debug["estimated_energy_confidence"] = round(float(estimated_confidence), 3)
        return result, debug

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

    disc_edge_only = (
        annular_diameter == 0.0
        and best_method == "radial-profile"
        and measured_px <= disc_diameter * 2.4
        and not contour_consistent_with_radial
    )
    no_zone_fallback_used = (
        measured_px <= disc_diameter * 1.02 and gradient_strength < 12.0 and edge_density < 0.035
    ) or disc_edge_only
    if no_zone_fallback_used:
        measured_px = disc_diameter

    review_required = (
        best_confidence < settings.MEASUREMENT_MIN_CONFIDENCE
        or radial_spread > disc_radius * 1.2
        or clipped_fraction >= 0.25
        or best_method not in {"radial-profile", "annular-baseline"}
    )
    warning = ""
    if no_zone_fallback_used:
        warning = "No visible inhibition zone gradient was found."
    elif radial_spread > disc_radius * 1.2:
        warning = "Zone edge is irregular across radial samples."
    elif clipped_fraction >= 0.25:
        warning = "Zone reaches the image border; diameter was estimated from visible arcs."
    elif best_confidence < settings.MEASUREMENT_MIN_CONFIDENCE:
        warning = "Zone confidence is low; confirm the boundary."
    elif best_method not in {"radial-profile", "annular-baseline"}:
        warning = "Zone boundary was recovered from contour fallback."

    result = {
        "diameter_px": round(max(measured_px, disc_diameter), 2),
        "confidence": round(max(0.0, min(0.97, best_confidence)), 3),
        "review_required": review_required,
        "no_zone_fallback_used": no_zone_fallback_used,
        "method": best_method,
        "warning": warning,
    }
    debug["overlay"] = _build_zone_overlay(roi, roi_cx, roi_cy, disc_radius, float(result["diameter_px"]))
    debug["selected_candidates"] = [
        {
            "diameter_px": round(float(diameter), 2),
            "confidence": round(float(confidence), 3),
            "method": method,
        }
        for diameter, confidence, method in candidates[:6]
    ]
    return result, debug


def measure_zone(
    image: np.ndarray,
    disc: Tuple[float, float, float],
    *,
    plate_detection: PlateDetection | None = None,
    extra_valid_mask: np.ndarray | None = None,
) -> Dict[str, float | bool | str]:
    result, _ = _measure_zone_internal(
        image,
        disc,
        plate_detection=plate_detection,
        extra_valid_mask=extra_valid_mask,
        return_debug=False,
    )
    return result


def measure_zone_with_debug(
    image: np.ndarray,
    disc: Tuple[float, float, float],
    *,
    plate_detection: PlateDetection | None = None,
    extra_valid_mask: np.ndarray | None = None,
) -> tuple[Dict[str, float | bool | str], Dict[str, object]]:
    return _measure_zone_internal(
        image,
        disc,
        plate_detection=plate_detection,
        extra_valid_mask=extra_valid_mask,
        return_debug=True,
    )


def detect_zones(
    image: np.ndarray,
    discs: List[Tuple[float, float, float]],
    *,
    plate_detection: PlateDetection | None = None,
    extra_valid_masks: Sequence[np.ndarray | None] | None = None,
) -> Tuple[List[float], List[float]]:
    diameters: List[float] = []
    confidences: List[float] = []
    for index, disc in enumerate(discs):
        extra_mask = None
        if extra_valid_masks is not None and index < len(extra_valid_masks):
            extra_mask = extra_valid_masks[index]
        result = measure_zone(image, disc, plate_detection=plate_detection, extra_valid_mask=extra_mask)
        diameters.append(float(result["diameter_px"]))
        confidences.append(float(result["confidence"]))
    return diameters, confidences
