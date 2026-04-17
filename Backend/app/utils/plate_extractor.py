from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from app.models.schemas import PlateDetection, Point

PlateCircle = Tuple[float, float, float, str]


def _resize_for_detection(image: np.ndarray, max_dimension: int = 1400) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_dimension:
        return image, 1.0

    scale = max_dimension / float(longest_side)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _circle_support_score(edge_map: np.ndarray, cx: float, cy: float, radius: float) -> float:
    if radius <= 0:
        return float("-inf")

    sample_count = max(180, int(radius * 3.2))
    angles = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False)
    ring_scores: list[float] = []
    support_hits: list[np.ndarray] = []

    for radius_offset in (-4.0, -2.0, 0.0, 2.0, 4.0):
        current_radius = max(1.0, radius + radius_offset)
        xs = np.clip(cx + np.cos(angles) * current_radius, 0, edge_map.shape[1] - 1).astype(int)
        ys = np.clip(cy + np.sin(angles) * current_radius, 0, edge_map.shape[0] - 1).astype(int)
        sampled = edge_map[ys, xs].astype(np.float32)
        ring_scores.append(float(np.mean(sampled)))
        support_hits.append(sampled > 0)

    coverage = float(np.mean(np.logical_or.reduce(support_hits))) if support_hits else 0.0
    edge_strength = float(np.mean(ring_scores)) if ring_scores else 0.0

    inner_radius = max(1.0, radius - 12.0)
    outer_radius = radius + 12.0
    inner_x = np.clip(cx + np.cos(angles) * inner_radius, 0, edge_map.shape[1] - 1).astype(int)
    inner_y = np.clip(cy + np.sin(angles) * inner_radius, 0, edge_map.shape[0] - 1).astype(int)
    outer_x = np.clip(cx + np.cos(angles) * outer_radius, 0, edge_map.shape[1] - 1).astype(int)
    outer_y = np.clip(cy + np.sin(angles) * outer_radius, 0, edge_map.shape[0] - 1).astype(int)
    ring_contrast = edge_strength - (
        (float(np.mean(edge_map[inner_y, inner_x])) + float(np.mean(edge_map[outer_y, outer_x]))) / 2.0
    )
    return (coverage * 180.0) + (edge_strength * 0.55) + (ring_contrast * 0.35)


def _generate_hough_candidates(gray: np.ndarray) -> list[PlateCircle]:
    min_dimension = min(gray.shape[:2])
    min_radius = max(60, int(min_dimension * 0.34))
    max_radius = max(min_radius + 16, int(min_dimension * 0.62))

    blurred = cv2.GaussianBlur(gray, (9, 9), 1.6)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blurred)

    candidates: list[PlateCircle] = []
    for source_name, source_image in (("hough-blur", blurred), ("hough-clahe", enhanced)):
        for accumulator_threshold in (32, 28, 24, 20):
            circles = cv2.HoughCircles(
                source_image,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(80, min_dimension // 2),
                param1=120,
                param2=accumulator_threshold,
                minRadius=min_radius,
                maxRadius=max_radius,
            )
            if circles is None:
                continue
            for x, y, radius in circles[0][:12]:
                candidates.append((float(x), float(y), float(radius), source_name))

    return candidates


def _detect_plate_from_hough(image: np.ndarray) -> Optional[PlateCircle]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    resized, scale = _resize_for_detection(gray)
    edge_map = cv2.Canny(cv2.GaussianBlur(resized, (9, 9), 1.4), 36, 108)
    candidates = _generate_hough_candidates(resized)
    if not candidates:
        return None

    best_candidate: Optional[PlateCircle] = None
    best_score = float("-inf")
    for cx, cy, radius, method in candidates:
        score = _circle_support_score(edge_map, cx, cy, radius)
        if score <= best_score:
            continue
        best_candidate = (cx, cy, radius, method)
        best_score = score

    if best_candidate is None:
        return None

    cx, cy, radius, method = best_candidate
    if scale != 1.0:
        inv_scale = 1.0 / scale
        return cx * inv_scale, cy * inv_scale, radius * inv_scale, method
    return best_candidate


def _detect_plate_from_contour(image: np.ndarray) -> Optional[PlateCircle]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blurred)
    edges = cv2.Canny(enhanced, 32, 92)
    edges = cv2.dilate(edges, np.ones((5, 5), dtype=np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = float(image.shape[0] * image.shape[1])
    candidates: list[PlateCircle] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.18:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = float((4.0 * np.pi * area) / (perimeter * perimeter))
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < min(image.shape[:2]) * 0.28:
            continue
        score = circularity + min(0.35, area / image_area)
        candidates.append((float(cx), float(cy), float(radius), f"contour:{score:.3f}"))

    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item[2]))


def detect_plate_circle(image: np.ndarray, yolo_bbox: Optional[tuple] = None) -> Optional[PlateCircle]:
    if image is None or image.size == 0:
        return None

    if yolo_bbox is not None:
        x1, y1, x2, y2 = yolo_bbox
        cx = float(x1 + x2) / 2.0
        cy = float(y1 + y2) / 2.0
        radius = float(max(x2 - x1, y2 - y1)) / 2.0
        return cx, cy, radius, "yolo-bbox"

    hough_circle = _detect_plate_from_hough(image)
    if hough_circle is not None:
        return hough_circle
    return _detect_plate_from_contour(image)


def build_plate_mask(
    image: np.ndarray,
    plate_circle: Optional[tuple[float, float, float]] = None,
    inset_ratio: float = 0.97,
) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    if plate_circle is None:
        detected = detect_plate_circle(image)
        if detected is None:
            return np.full(image.shape[:2], 255, dtype=np.uint8)
        cx, cy, radius, _ = detected
    else:
        cx, cy, radius = plate_circle

    usable_radius = max(1, int(round(radius * inset_ratio)))
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), usable_radius, 255, -1)
    return mask


def extract_plate(
    image: np.ndarray,
    yolo_bbox: Optional[tuple] = None,
) -> Tuple[np.ndarray, Optional[PlateDetection]]:
    if image is None or image.size == 0:
        return image, None

    detected = detect_plate_circle(image, yolo_bbox=yolo_bbox)
    if detected is None:
        return image, None

    cx, cy, radius, method = detected
    height, width = image.shape[:2]
    margin = int(round(radius * 1.03))
    x1 = max(0, int(round(cx - margin)))
    y1 = max(0, int(round(cy - margin)))
    x2 = min(width, int(round(cx + margin)))
    y2 = min(height, int(round(cy + margin)))

    cropped = image[y1:y2, x1:x2].copy()
    if cropped.size == 0:
        cropped = image
        x1 = 0
        y1 = 0
        x2 = width
        y2 = height

    clipped_pixels = 0
    if x1 == 0:
        clipped_pixels += int(max(0.0, margin - cx))
    if y1 == 0:
        clipped_pixels += int(max(0.0, margin - cy))
    if x2 == width:
        clipped_pixels += int(max(0.0, (cx + margin) - width))
    if y2 == height:
        clipped_pixels += int(max(0.0, (cy + margin) - height))

    clip_fraction = float(clipped_pixels) / max(1.0, radius * 4.0)
    warnings: list[str] = []
    clipped = clip_fraction > 0.025
    if clipped:
        warnings.append("Plate touches the image border.")

    detection = PlateDetection(
        center=Point(x=round(cx, 2), y=round(cy, 2)),
        radius_px=round(radius, 2),
        clipped=clipped,
        clip_fraction=round(clip_fraction, 4),
        method=method,
        warnings=warnings,
    )
    return cropped, detection


def extract_plate_roi(image: np.ndarray, yolo_bbox: Optional[tuple] = None) -> np.ndarray:
    cropped, _ = extract_plate(image, yolo_bbox=yolo_bbox)
    return cropped
