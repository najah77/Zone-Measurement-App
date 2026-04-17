from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.utils.plate_extractor import build_plate_mask, detect_plate_circle
from app.utils.preprocessing import preprocess_plate

DiscCircle = Tuple[float, float, float]


def validate_disc_geometry(discs: Sequence[DiscCircle]) -> bool:
    if len(discs) < 3:
        return False
    radii = np.array([radius for _, _, radius in discs], dtype=float)
    mean_radius = float(np.mean(radii))
    if mean_radius <= 0:
        return False
    return float(np.std(radii)) <= mean_radius * 0.32


def _merge_similar_discs(discs: Sequence[DiscCircle]) -> List[DiscCircle]:
    merged: List[DiscCircle] = []
    for candidate in sorted(discs, key=lambda disc: disc[2], reverse=True):
        if any(
            np.hypot(candidate[0] - kept[0], candidate[1] - kept[1]) < max(candidate[2], kept[2]) * 0.9
            for kept in merged
        ):
            continue
        merged.append(candidate)
    return merged


def _plate_disc_radius_bounds(image: np.ndarray) -> tuple[float, float, np.ndarray]:
    plate = detect_plate_circle(image)
    if plate is None:
        base = min(image.shape[:2])
        min_radius = max(12.0, base * 0.022)
        max_radius = max(min_radius + 10.0, base * 0.07)
        return min_radius, max_radius, np.full(image.shape[:2], 255, dtype=np.uint8)

    cx, cy, plate_radius, _ = plate
    min_radius = max(12.0, plate_radius * 0.03)
    max_radius = max(min_radius + 8.0, plate_radius * 0.095)
    plate_mask = build_plate_mask(image, plate_circle=(cx, cy, plate_radius), inset_ratio=0.965)
    return min_radius, max_radius, plate_mask


def _white_disc_mask(image: np.ndarray, plate_mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV) if hsv.ndim == 3 and image.ndim == 2 else hsv
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(lab, cv2.COLOR_BGR2LAB) if lab.ndim == 3 and image.ndim == 2 else lab

    value_channel = hsv[:, :, 2]
    saturation_channel = hsv[:, :, 1]
    lightness_channel = lab[:, :, 0]
    inside_plate = plate_mask > 0

    value_threshold = max(145, int(np.percentile(value_channel[inside_plate], 97.0))) if np.any(inside_plate) else 145
    lightness_threshold = max(
        145,
        int(np.percentile(lightness_channel[inside_plate], 97.0)),
    ) if np.any(inside_plate) else 145
    saturation_threshold = 115

    hsv_mask = cv2.inRange(hsv, (0, 0, value_threshold), (180, saturation_threshold, 255))
    light_mask = cv2.threshold(lightness_channel, lightness_threshold, 255, cv2.THRESH_BINARY)[1]
    local_highlights = cv2.morphologyEx(
        lightness_channel,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    )
    highlight_threshold = max(18, int(np.percentile(local_highlights[inside_plate], 92.0))) if np.any(inside_plate) else 18
    highlight_mask = cv2.threshold(local_highlights, highlight_threshold, 255, cv2.THRESH_BINARY)[1]

    combined = cv2.bitwise_or(hsv_mask, light_mask)
    combined = cv2.bitwise_or(combined, highlight_mask)
    combined = cv2.bitwise_and(combined, combined, mask=plate_mask)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    return combined


def _core_disc_mask(image: np.ndarray, plate_mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV) if hsv.ndim == 3 and image.ndim == 2 else hsv
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(lab, cv2.COLOR_BGR2LAB) if lab.ndim == 3 and image.ndim == 2 else lab

    inside_plate = plate_mask > 0
    value_channel = hsv[:, :, 2]
    saturation_channel = hsv[:, :, 1]
    lightness_channel = lab[:, :, 0]

    if np.any(inside_plate):
        core_value = min(254, max(170, int(np.percentile(value_channel[inside_plate], 99.3))))
        core_lightness = min(254, max(170, int(np.percentile(lightness_channel[inside_plate], 99.3))))
    else:
        core_value = 170
        core_lightness = 170

    value_mask = cv2.threshold(value_channel, core_value, 255, cv2.THRESH_BINARY)[1]
    light_mask = cv2.threshold(lightness_channel, core_lightness, 255, cv2.THRESH_BINARY)[1]
    core_mask = cv2.bitwise_or(value_mask, light_mask)
    saturation_limit = cv2.threshold(saturation_channel, 120, 255, cv2.THRESH_BINARY_INV)[1]
    core_mask = cv2.bitwise_and(core_mask, saturation_limit)
    core_mask = cv2.bitwise_and(core_mask, core_mask, mask=plate_mask)
    core_mask = cv2.morphologyEx(core_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    core_mask = cv2.morphologyEx(core_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    return core_mask


def _component_disc_candidates(
    image: np.ndarray,
    white_mask: np.ndarray,
    min_radius: float,
    max_radius: float,
) -> List[DiscCircle]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    discs: List[DiscCircle] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 0:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = float((4.0 * np.pi * area) / (perimeter * perimeter))
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        equivalent_radius = float(np.sqrt(area / np.pi))
        refined_radius = float(np.median([radius, equivalent_radius]))

        if not (min_radius * 0.75 <= refined_radius <= max_radius * 1.2):
            continue
        if circularity < 0.62:
            continue

        roi_radius = int(max(refined_radius * 1.65, refined_radius + 18.0))
        x1 = max(0, int(round(cx)) - roi_radius)
        y1 = max(0, int(round(cy)) - roi_radius)
        x2 = min(gray.shape[1], int(round(cx)) + roi_radius)
        y2 = min(gray.shape[0], int(round(cy)) + roi_radius)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        yy, xx = np.indices(roi.shape)
        local_cx = float(cx) - x1
        local_cy = float(cy) - y1
        distances = np.sqrt((xx - local_cx) ** 2 + (yy - local_cy) ** 2)
        inner = roi[distances <= refined_radius * 0.72]
        annulus = roi[(distances >= refined_radius * 0.95) & (distances <= refined_radius * 1.55)]
        if inner.size == 0 or annulus.size == 0:
            continue

        center_brightness = float(np.mean(inner))
        ring_brightness = float(np.mean(annulus))
        if center_brightness < 132 or (center_brightness - ring_brightness) < 10.0:
            continue

        discs.append((float(cx), float(cy), float(refined_radius)))

    return discs


def _hough_disc_candidates(
    image: np.ndarray,
    plate_mask: np.ndarray,
    min_radius: float,
    max_radius: float,
) -> List[DiscCircle]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    masked_gray = gray.copy()
    masked_gray[plate_mask == 0] = int(np.median(gray))
    top_hat = cv2.morphologyEx(
        masked_gray,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(7, int(round(max_radius * 1.4))),) * 2),
    )
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(top_hat, (5, 5), 0),
        cv2.HOUGH_GRADIENT,
        dp=1.15,
        minDist=max(18, int(round(min_radius * 2.8))),
        param1=110,
        param2=12,
        minRadius=max(8, int(round(min_radius * 0.8))),
        maxRadius=max(10, int(round(max_radius * 1.15))),
    )
    if circles is None:
        return []

    discs: List[DiscCircle] = []
    for x, y, radius in circles[0]:
        ix = int(np.clip(round(x), 0, plate_mask.shape[1] - 1))
        iy = int(np.clip(round(y), 0, plate_mask.shape[0] - 1))
        if plate_mask[iy, ix] == 0:
            continue
        discs.append((float(x), float(y), float(radius)))
    return discs


def _refine_disc_radius(image: np.ndarray, disc: DiscCircle) -> DiscCircle:
    x, y, radius = disc
    if radius <= 0:
        return disc

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    roi_radius = int(max(radius * 1.6, radius + 18))
    x1 = max(0, int(round(x)) - roi_radius)
    y1 = max(0, int(round(y)) - roi_radius)
    x2 = min(gray.shape[1], int(round(x)) + roi_radius)
    y2 = min(gray.shape[0], int(round(y)) + roi_radius)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return disc

    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    local_center = np.array([roi.shape[1] / 2.0, roi.shape[0] / 2.0], dtype=float)
    best_radius = radius
    best_score = float("-inf")
    for percentile in (95, 96, 97, 98, 99):
        threshold_value = max(150, int(np.percentile(blurred, percentile)))
        binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < np.pi * max(6.0, radius * 0.38) ** 2:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = float((4.0 * np.pi * area) / (perimeter * perimeter))
            (cx, cy), candidate_radius = cv2.minEnclosingCircle(contour)
            if not (radius * 0.55 <= candidate_radius <= radius * 1.2):
                continue
            center_offset = float(np.linalg.norm(np.array([cx, cy], dtype=float) - local_center))

            contour_mask = np.zeros_like(roi, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            mean_brightness = float(np.mean(roi[contour_mask > 0])) if np.any(contour_mask > 0) else 0.0
            radius_penalty = abs(candidate_radius - (radius * 0.9)) * 4.0
            score = (circularity * 110.0) + (mean_brightness * 0.18) - (center_offset * 3.5) - radius_penalty
            if score > best_score:
                best_score = score
                best_radius = float(candidate_radius)

    return float(x), float(y), float(best_radius)


def refine_detected_discs(image: np.ndarray, discs: Sequence[DiscCircle]) -> List[DiscCircle]:
    return [_refine_disc_radius(image, disc) for disc in discs]


def detect_discs(image: np.ndarray) -> List[DiscCircle]:
    if image is None or image.size == 0:
        return []

    base_image = image.copy()
    preprocessed = preprocess_plate(image)
    min_radius, max_radius, plate_mask = _plate_disc_radius_bounds(base_image)
    white_mask = _white_disc_mask(base_image, plate_mask)
    core_mask = _core_disc_mask(base_image, plate_mask)

    component_discs = _merge_similar_discs(
        _component_disc_candidates(base_image, white_mask, min_radius, max_radius)
        + _component_disc_candidates(base_image, core_mask, min_radius, max_radius)
    )
    if len(component_discs) >= 3 and validate_disc_geometry(component_discs):
        candidates = _merge_similar_discs(component_discs)
    else:
        hough_discs = _hough_disc_candidates(preprocessed, plate_mask, min_radius, max_radius)
        candidates = _merge_similar_discs(component_discs + hough_discs)

    if not candidates:
        return []

    refined = refine_detected_discs(base_image, candidates)
    radii = np.array([radius for _, _, radius in refined], dtype=float)
    baseline_radius = float(np.median(radii))
    filtered = [
        disc
        for disc in refined
        if baseline_radius * 0.7 <= disc[2] <= baseline_radius * 1.35
        and plate_mask[
            int(np.clip(round(disc[1]), 0, plate_mask.shape[0] - 1)),
            int(np.clip(round(disc[0]), 0, plate_mask.shape[1] - 1)),
        ] > 0
    ]

    filtered = _merge_similar_discs(filtered)
    filtered = sorted(filtered, key=lambda disc: (round(disc[1] / max(40.0, baseline_radius * 1.8)), disc[0]))
    return filtered[: settings.MAX_DISCS]
