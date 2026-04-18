from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.models.schemas import PlateDetection
from app.utils.plate_extractor import build_plate_mask, detect_plate_circle
from app.utils.preprocessing import preprocess_plate

DiscCircle = Tuple[float, float, float]


def _round_metrics(metrics: Dict[str, float] | None) -> Dict[str, float]:
    if not metrics:
        return {}
    return {key: round(float(value), 4) for key, value in metrics.items()}


def _candidate_record(
    disc: DiscCircle,
    *,
    source: str,
    stage: str,
    score: float | None = None,
    reason: str | None = None,
    metrics: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "x": round(float(disc[0]), 2),
        "y": round(float(disc[1]), 2),
        "radius": round(float(disc[2]), 2),
        "source": source,
        "stage": stage,
    }
    if score is not None:
        payload["score"] = round(float(score), 2)
    if reason:
        payload["reason"] = reason
    rounded_metrics = _round_metrics(metrics)
    if rounded_metrics:
        payload["metrics"] = rounded_metrics
    return payload


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


def _disc_candidate_metrics(
    gray: np.ndarray,
    white_mask: np.ndarray,
    core_mask: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
) -> Dict[str, float] | None:
    if radius <= 0:
        return None

    roi_radius = int(max(radius * 1.8, radius + 18.0))
    x1 = max(0, int(round(cx)) - roi_radius)
    y1 = max(0, int(round(cy)) - roi_radius)
    x2 = min(gray.shape[1], int(round(cx)) + roi_radius)
    y2 = min(gray.shape[0], int(round(cy)) + roi_radius)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    yy, xx = np.indices(roi.shape)
    local_cx = float(cx) - x1
    local_cy = float(cy) - y1
    distances = np.sqrt((xx - local_cx) ** 2 + (yy - local_cy) ** 2)
    inner_mask = distances <= radius * 0.72
    disc_mask = distances <= radius * 0.95
    annulus_mask = (distances >= radius * 1.05) & (distances <= radius * 1.6)
    if np.count_nonzero(inner_mask) < 10 or np.count_nonzero(annulus_mask) < 10:
        return None

    inner_values = roi[inner_mask]
    annulus_values = roi[annulus_mask]
    white_values = white_mask[y1:y2, x1:x2]
    core_values = core_mask[y1:y2, x1:x2]
    return {
        "center_mean": float(np.mean(inner_values)),
        "center_std": float(np.std(inner_values)),
        "bright_fraction": float(np.mean(inner_values >= 170)),
        "white_occupancy": float(np.mean(white_values[disc_mask] > 0)),
        "core_occupancy": float(np.mean(core_values[disc_mask] > 0)),
        "annulus_mean": float(np.mean(annulus_values)),
        "contrast": float(np.mean(inner_values) - np.mean(annulus_values)),
    }


def _disc_candidate_score(metrics: Dict[str, float], radius: float, min_radius: float, max_radius: float) -> float:
    radius_midpoint = (min_radius + max_radius) * 0.5
    radius_penalty = abs(radius - radius_midpoint) * 1.8
    score = 0.0
    score += max(0.0, metrics["center_mean"] - 180.0) * 1.2
    score += max(0.0, metrics["contrast"] - 18.0) * 1.7
    score += metrics["bright_fraction"] * 85.0
    score += metrics["white_occupancy"] * 130.0
    score += metrics["core_occupancy"] * 65.0
    score -= min(75.0, metrics["center_std"] * 0.75)
    score -= radius_penalty
    return float(score)


def _plate_disc_radius_bounds(
    image: np.ndarray,
    plate_detection: PlateDetection | None = None,
) -> tuple[float, float, np.ndarray]:
    if plate_detection and plate_detection.shape == "rectangle":
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        inset_x = max(8, int(round(width * 0.025)))
        inset_y = max(8, int(round(height * 0.025)))
        cv2.rectangle(mask, (inset_x, inset_y), (max(inset_x + 1, width - inset_x), max(inset_y + 1, height - inset_y)), 255, -1)
        base = min(height, width)
        min_radius = max(10.0, base * 0.018)
        max_radius = max(min_radius + 8.0, base * 0.058)
        return min_radius, max_radius, mask

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
    core_mask: np.ndarray,
    min_radius: float,
    max_radius: float,
    *,
    dense_layout: bool = False,
) -> List[DiscCircle]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    solid_mask = cv2.bitwise_or(
        white_mask,
        cv2.dilate(core_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1),
    )
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    solid_mask = cv2.morphologyEx(solid_mask, cv2.MORPH_CLOSE, close_kernel)
    contours, _ = cv2.findContours(solid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

        if not _is_bright_disc_center(
            gray,
            white_mask,
            core_mask,
            float(cx),
            float(cy),
            refined_radius,
            dense_layout=dense_layout,
        ):
            continue

        discs.append((float(cx), float(cy), float(refined_radius)))

    return discs


def _hough_disc_candidates(
    image: np.ndarray,
    plate_mask: np.ndarray,
    white_mask: np.ndarray,
    core_mask: np.ndarray,
    min_radius: float,
    max_radius: float,
    prefer_dense_layout: bool = False,
) -> List[DiscCircle]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    masked_gray = gray.copy()
    masked_gray[plate_mask == 0] = int(np.median(gray))

    discs: List[DiscCircle] = []
    sources = []
    top_hat = cv2.morphologyEx(
        masked_gray,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(7, int(round(max_radius * 1.4))),) * 2),
    )
    sources.append(cv2.GaussianBlur(top_hat, (5, 5), 0))
    if prefer_dense_layout:
        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(masked_gray)
        sources.append(cv2.GaussianBlur(clahe, (5, 5), 0))

    for source in sources:
        circles = cv2.HoughCircles(
            source,
            cv2.HOUGH_GRADIENT,
            dp=1.15,
            minDist=max(18, int(round(min_radius * (2.3 if prefer_dense_layout else 2.8)))),
            param1=90 if prefer_dense_layout else 110,
            param2=20 if prefer_dense_layout else 12,
            minRadius=max(8, int(round(min_radius * 0.8))),
            maxRadius=max(10, int(round(max_radius * 1.15))),
        )
        if circles is None:
            continue

        for x, y, radius in circles[0]:
            ix = int(np.clip(round(x), 0, plate_mask.shape[1] - 1))
            iy = int(np.clip(round(y), 0, plate_mask.shape[0] - 1))
            if plate_mask[iy, ix] == 0:
                continue
            if not _is_bright_disc_center(
                gray,
                white_mask,
                core_mask,
                float(x),
                float(y),
                float(radius),
                dense_layout=prefer_dense_layout,
            ):
                continue
            discs.append((float(x), float(y), float(radius)))
    return discs


def _blob_disc_candidates(
    image: np.ndarray,
    plate_mask: np.ndarray,
    white_mask: np.ndarray,
    core_mask: np.ndarray,
    min_radius: float,
    max_radius: float,
) -> List[DiscCircle]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    masked_gray = gray.copy()
    masked_gray[plate_mask == 0] = int(np.median(gray))

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.minThreshold = 140
    params.maxThreshold = 255
    params.thresholdStep = 5
    params.filterByArea = True
    params.minArea = float(np.pi * (min_radius * 0.55) ** 2)
    params.maxArea = float(np.pi * (max_radius * 1.25) ** 2)
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.minDistBetweenBlobs = max(18.0, min_radius * 1.4)

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(masked_gray)
    discs: List[DiscCircle] = []
    for keypoint in keypoints:
        radius = float(keypoint.size / 2.0)
        if not (min_radius * 0.65 <= radius <= max_radius * 1.15):
            continue
        x = float(keypoint.pt[0])
        y = float(keypoint.pt[1])
        if not _is_bright_disc_center(gray, white_mask, core_mask, x, y, radius, dense_layout=True):
            continue
        discs.append((x, y, radius))
    return discs


def _supplemental_phone_hough_candidates(
    image: np.ndarray,
    plate_mask: np.ndarray,
    white_mask: np.ndarray,
    core_mask: np.ndarray,
    min_radius: float,
    max_radius: float,
) -> List[Tuple[DiscCircle, str]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    masked_gray = gray.copy()
    masked_gray[plate_mask == 0] = int(np.median(gray))
    white_source = cv2.GaussianBlur(white_mask, (9, 9), 0)
    gray_source = cv2.GaussianBlur(masked_gray, (9, 9), 0)

    candidates: List[Tuple[DiscCircle, str]] = []
    for source_name, source, param1, param2 in (
        ("hough_gray", gray_source, 80, 14),
        ("hough_white", white_source, 90, 14),
    ):
        circles = cv2.HoughCircles(
            source,
            cv2.HOUGH_GRADIENT,
            dp=1.1,
            minDist=max(22, int(round(min_radius * 2.2))),
            param1=param1,
            param2=param2,
            minRadius=max(8, int(round(min_radius * 0.7))),
            maxRadius=max(10, int(round(max_radius * 1.1))),
        )
        if circles is None:
            continue

        for x, y, radius in circles[0]:
            ix = int(np.clip(round(x), 0, plate_mask.shape[1] - 1))
            iy = int(np.clip(round(y), 0, plate_mask.shape[0] - 1))
            if plate_mask[iy, ix] == 0:
                continue

            metrics = _disc_candidate_metrics(gray, white_mask, core_mask, float(x), float(y), float(radius))
            if metrics is None:
                continue
            if metrics["contrast"] < 25.0:
                continue
            if metrics["center_mean"] < 145.0 and metrics["white_occupancy"] < 0.34:
                continue
            if metrics["bright_fraction"] < 0.32 and metrics["white_occupancy"] < 0.18:
                continue
            candidates.append(((float(x), float(y), float(radius)), source_name))

    return candidates


def _is_bright_disc_center_basic(
    gray: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
) -> bool:
    roi_radius = int(max(radius * 1.65, radius + 18.0))
    x1 = max(0, int(round(cx)) - roi_radius)
    y1 = max(0, int(round(cy)) - roi_radius)
    x2 = min(gray.shape[1], int(round(cx)) + roi_radius)
    y2 = min(gray.shape[0], int(round(cy)) + roi_radius)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    yy, xx = np.indices(roi.shape)
    local_cx = float(cx) - x1
    local_cy = float(cy) - y1
    distances = np.sqrt((xx - local_cx) ** 2 + (yy - local_cy) ** 2)
    inner = roi[distances <= radius * 0.72]
    annulus = roi[(distances >= radius * 0.95) & (distances <= radius * 1.55)]
    if inner.size == 0 or annulus.size == 0:
        return False

    center_brightness = float(np.mean(inner))
    ring_brightness = float(np.mean(annulus))
    bright_fraction = float(np.mean(inner >= 170))
    return (
        center_brightness >= 150
        and bright_fraction >= 0.45
        and (center_brightness - ring_brightness) >= 12.0
    )


def _is_bright_disc_center(
    gray: np.ndarray,
    white_mask: np.ndarray,
    core_mask: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    *,
    dense_layout: bool = False,
) -> bool:
    if not dense_layout:
        return _is_bright_disc_center_basic(gray, cx, cy, radius)
    metrics = _disc_candidate_metrics(gray, white_mask, core_mask, cx, cy, radius)
    if metrics is None:
        return False
    return (
        metrics["center_mean"] >= 188.0
        and metrics["bright_fraction"] >= 0.78
        and metrics["contrast"] >= 28.0
        and metrics["white_occupancy"] >= 0.18
        and (metrics["core_occupancy"] >= 0.08 or metrics["white_occupancy"] >= 0.34)
    )


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


def _merge_scored_candidates(
    candidates: Sequence[Tuple[DiscCircle, float, str, Dict[str, float]]],
) -> tuple[List[Tuple[DiscCircle, float, str, Dict[str, float]]], List[Dict[str, Any]]]:
    accepted: List[Tuple[DiscCircle, float, str, Dict[str, float]]] = []
    rejected: List[Dict[str, Any]] = []
    for disc, score, source, metrics in sorted(candidates, key=lambda item: item[1], reverse=True):
        duplicate_of = next(
            (
                kept
                for kept in accepted
                if np.hypot(disc[0] - kept[0][0], disc[1] - kept[0][1]) < max(disc[2], kept[0][2]) * 0.9
            ),
            None,
        )
        if duplicate_of is not None:
            rejected.append(
                _candidate_record(
                    disc,
                    source=source,
                    stage="rejected",
                    score=score,
                    reason=f"duplicate_of_{duplicate_of[2]}",
                    metrics=metrics,
                )
            )
            continue
        accepted.append((disc, score, source, metrics))
    return accepted, rejected


def detect_discs_with_debug(
    image: np.ndarray,
    plate_detection: PlateDetection | None = None,
) -> tuple[List[DiscCircle], dict]:
    if image is None or image.size == 0:
        return [], {}

    base_image = image.copy()
    preprocessed = preprocess_plate(image)
    min_radius, max_radius, plate_mask = _plate_disc_radius_bounds(base_image, plate_detection=plate_detection)
    white_mask = _white_disc_mask(base_image, plate_mask)
    core_mask = _core_disc_mask(base_image, plate_mask)
    dense_layout = bool(plate_detection and plate_detection.shape == "rectangle")

    component_candidates = _component_disc_candidates(
        base_image,
        white_mask,
        core_mask,
        min_radius,
        max_radius,
        dense_layout=dense_layout,
    ) + _component_disc_candidates(
        base_image,
        core_mask,
        core_mask,
        min_radius,
        max_radius,
        dense_layout=dense_layout,
    )
    component_discs = _merge_similar_discs(component_candidates)
    hough_discs: List[DiscCircle] = []
    blob_discs: List[DiscCircle] = []
    raw_candidates: List[Tuple[DiscCircle, str]] = [(disc, "component") for disc in component_discs]
    fallback_detection_used = False
    suspicious_low_count = False
    if dense_layout:
        blob_discs = _blob_disc_candidates(base_image, plate_mask, white_mask, core_mask, min_radius, max_radius)
        hough_discs = _hough_disc_candidates(
            preprocessed,
            plate_mask,
            white_mask,
            core_mask,
            min_radius,
            max_radius,
            prefer_dense_layout=True,
        )
        raw_candidates.extend((disc, "blob") for disc in blob_discs)
        raw_candidates.extend((disc, "hough_dense") for disc in hough_discs)
    else:
        suspicious_low_count = len(component_discs) < 6
        geometry_valid = len(component_discs) >= 3 and validate_disc_geometry(component_discs)
        if not geometry_valid:
            standard_hough = _hough_disc_candidates(
                preprocessed,
                plate_mask,
                white_mask,
                core_mask,
                min_radius,
                max_radius,
            )
            hough_discs.extend(standard_hough)
            raw_candidates.extend((disc, "hough_standard") for disc in standard_hough)
        if suspicious_low_count:
            fallback_detection_used = True
            supplemental_hough = _supplemental_phone_hough_candidates(
                base_image,
                plate_mask,
                white_mask,
                core_mask,
                min_radius,
                max_radius,
            )
            hough_discs.extend(disc for disc, _ in supplemental_hough)
            raw_candidates.extend(supplemental_hough)

    if not raw_candidates:
        return [], {
            "preprocessed": preprocessed,
            "plate_mask": plate_mask,
            "white_mask": white_mask,
            "core_mask": core_mask,
            "component_candidates": component_candidates,
            "blob_candidates": blob_discs,
            "hough_candidates": hough_discs,
            "filtered_candidates": [],
            "raw_candidate_records": [],
            "accepted_candidates": [],
            "rejected_candidates": [],
            "fallback_detection_used": fallback_detection_used,
            "suspicious_low_count": suspicious_low_count,
            "radius_bounds": {"min_radius": round(min_radius, 2), "max_radius": round(max_radius, 2)},
        }

    gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY) if base_image.ndim == 3 else base_image.copy()
    scored_candidates: List[Tuple[DiscCircle, float, str, Dict[str, float]]] = []
    rejected_candidates: List[Dict[str, Any]] = []
    raw_candidate_records = [_candidate_record(disc, source=source, stage="raw") for disc, source in raw_candidates]

    for raw_disc, source in raw_candidates:
        disc = _refine_disc_radius(base_image, raw_disc)
        ix = int(np.clip(round(disc[0]), 0, plate_mask.shape[1] - 1))
        iy = int(np.clip(round(disc[1]), 0, plate_mask.shape[0] - 1))
        if plate_mask[iy, ix] == 0:
            rejected_candidates.append(
                _candidate_record(disc, source=source, stage="rejected", reason="outside_plate_mask")
            )
            continue

        metrics = _disc_candidate_metrics(gray, white_mask, core_mask, disc[0], disc[1], disc[2])
        if metrics is None:
            rejected_candidates.append(
                _candidate_record(disc, source=source, stage="rejected", reason="metrics_unavailable")
            )
            continue

        if dense_layout:
            if metrics.get("white_occupancy", 0.0) < 0.16:
                rejected_candidates.append(
                    _candidate_record(
                        disc,
                        source=source,
                        stage="rejected",
                        reason="low_white_occupancy",
                        metrics=metrics,
                    )
                )
                continue
            score = _disc_candidate_score(metrics, disc[2], min_radius, max_radius)
            if score < 155.0:
                rejected_candidates.append(
                    _candidate_record(
                        disc,
                        source=source,
                        stage="rejected",
                        score=score,
                        reason="low_disc_score",
                        metrics=metrics,
                    )
                )
                continue
        else:
            if metrics.get("contrast", 0.0) < 18.0:
                rejected_candidates.append(
                    _candidate_record(
                        disc,
                        source=source,
                        stage="rejected",
                        reason="low_contrast",
                        metrics=metrics,
                    )
                )
                continue
            if metrics.get("bright_fraction", 0.0) < 0.18 and metrics.get("white_occupancy", 0.0) < 0.16:
                rejected_candidates.append(
                    _candidate_record(
                        disc,
                        source=source,
                        stage="rejected",
                        reason="low_disc_signal",
                        metrics=metrics,
                    )
                )
                continue
            if metrics.get("center_mean", 0.0) < 145.0 and metrics.get("white_occupancy", 0.0) < 0.34:
                rejected_candidates.append(
                    _candidate_record(
                        disc,
                        source=source,
                        stage="rejected",
                        reason="low_center_brightness",
                        metrics=metrics,
                    )
                )
                continue
            score = _disc_candidate_score(metrics, disc[2], min_radius, max_radius)
            if score < 95.0:
                rejected_candidates.append(
                    _candidate_record(
                        disc,
                        source=source,
                        stage="rejected",
                        score=score,
                        reason="low_disc_score",
                        metrics=metrics,
                    )
                )
                continue

        scored_candidates.append((disc, score, source, metrics))

    if not scored_candidates:
        return [], {
            "preprocessed": preprocessed,
            "plate_mask": plate_mask,
            "white_mask": white_mask,
            "core_mask": core_mask,
            "component_candidates": component_candidates,
            "blob_candidates": blob_discs,
            "hough_candidates": hough_discs,
            "filtered_candidates": [],
            "raw_candidate_records": raw_candidate_records,
            "accepted_candidates": [],
            "rejected_candidates": rejected_candidates,
            "fallback_detection_used": fallback_detection_used,
            "suspicious_low_count": suspicious_low_count,
            "radius_bounds": {"min_radius": round(min_radius, 2), "max_radius": round(max_radius, 2)},
        }

    merged_candidates, duplicate_rejections = _merge_scored_candidates(scored_candidates)
    rejected_candidates.extend(duplicate_rejections)

    baseline_pool = [disc[2] for disc, _, _, _ in merged_candidates[: max(4, min(len(merged_candidates), 10))]]
    baseline_radius = float(np.median(np.array(baseline_pool, dtype=float))) if baseline_pool else float(min_radius)

    filtered_candidates: List[Tuple[DiscCircle, float, str, Dict[str, float]]] = []
    for disc, score, source, metrics in merged_candidates:
        if not (baseline_radius * 0.68 <= disc[2] <= baseline_radius * 1.32):
            rejected_candidates.append(
                _candidate_record(
                    disc,
                    source=source,
                    stage="rejected",
                    score=score,
                    reason="radius_outlier",
                    metrics=metrics,
                )
            )
            continue
        if dense_layout:
            height, width = plate_mask.shape[:2]
            edge_margin = max(22.0, disc[2] * 2.1)
            if not (
                disc[0] >= edge_margin
                and disc[1] >= edge_margin
                and (width - disc[0]) >= edge_margin
                and (height - disc[1]) >= edge_margin
            ):
                rejected_candidates.append(
                    _candidate_record(
                        disc,
                        source=source,
                        stage="rejected",
                        score=score,
                        reason="edge_clipped_candidate",
                        metrics=metrics,
                    )
                )
                continue
        filtered_candidates.append((disc, score, source, metrics))

    filtered_candidates = filtered_candidates[: settings.MAX_DISCS]
    final_discs = [
        disc
        for disc, _, _, _ in sorted(
            filtered_candidates,
            key=lambda item: (round(item[0][1] / max(40.0, baseline_radius * 1.8)), item[0][0]),
        )
    ]
    return final_discs, {
        "preprocessed": preprocessed,
        "plate_mask": plate_mask,
        "white_mask": white_mask,
        "core_mask": core_mask,
        "component_candidates": component_candidates,
        "blob_candidates": blob_discs,
        "hough_candidates": hough_discs,
        "filtered_candidates": final_discs,
        "raw_candidate_records": raw_candidate_records,
        "accepted_candidates": [
            _candidate_record(disc, source=source, stage="accepted", score=score, metrics=metrics)
            for disc, score, source, metrics in filtered_candidates
        ],
        "rejected_candidates": rejected_candidates,
        "fallback_detection_used": fallback_detection_used,
        "suspicious_low_count": suspicious_low_count,
        "candidate_scores": [
            _candidate_record(disc, source=source, stage="scored", score=score, metrics=metrics)
            for disc, score, source, metrics in merged_candidates[: settings.MAX_DISCS + 12]
        ],
        "radius_bounds": {"min_radius": round(min_radius, 2), "max_radius": round(max_radius, 2)},
    }


def detect_discs(
    image: np.ndarray,
    plate_detection: PlateDetection | None = None,
) -> List[DiscCircle]:
    discs, _ = detect_discs_with_debug(image, plate_detection=plate_detection)
    return discs
