from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.utils.preprocessing import preprocess_plate


DiscCircle = Tuple[float, float, float]


def validate_disc_geometry(discs: Sequence[DiscCircle]) -> bool:
    if len(discs) < 3:
        return False
    radii = np.array([radius for _, _, radius in discs], dtype=float)
    mean_radius = float(np.mean(radii))
    if mean_radius <= 0:
        return False
    return float(np.std(radii)) <= mean_radius * 0.25


def is_valid_disc(image: np.ndarray, x: float, y: float, r: float) -> bool:
    h, w = image.shape[:2]
    x1, y1 = int(max(0, x - r * 0.6)), int(max(0, y - r * 0.6))
    x2, y2 = int(min(w, x + r * 0.6)), int(min(h, y + r * 0.6))
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    center_mean = float(np.mean(gray))

    ring_outer = int(max(2, round(r * 1.2)))
    rx1, ry1 = int(max(0, x - ring_outer)), int(max(0, y - ring_outer))
    rx2, ry2 = int(min(w, x + ring_outer)), int(min(h, y + ring_outer))
    outer_roi = image[ry1:ry2, rx1:rx2]
    if outer_roi.size == 0:
        return False
    outer_gray = cv2.cvtColor(outer_roi, cv2.COLOR_BGR2GRAY) if outer_roi.ndim == 3 else outer_roi
    yy, xx = np.indices(outer_gray.shape)
    cx = outer_gray.shape[1] / 2.0
    cy = outer_gray.shape[0] / 2.0
    distances = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    annulus_mask = (distances >= r * 0.65) & (distances <= r * 1.15)
    annulus_mean = float(np.mean(outer_gray[annulus_mask])) if np.any(annulus_mask) else center_mean

    return center_mean >= 170 and (center_mean - annulus_mean) >= 8


def _merge_similar_discs(discs: Sequence[DiscCircle]) -> List[DiscCircle]:
    merged: List[DiscCircle] = []
    for candidate in sorted(discs, key=lambda disc: disc[2], reverse=True):
        if any(np.hypot(candidate[0] - kept[0], candidate[1] - kept[1]) < max(candidate[2], kept[2]) * 0.8 for kept in merged):
            continue
        merged.append(candidate)
    return merged


def _detect_with_hough(gray: np.ndarray) -> List[DiscCircle]:
    min_radius = max(8, int(min(gray.shape[:2]) * 0.012))
    max_radius = max(min_radius + 5, int(min(gray.shape[:2]) * 0.05))
    top_hat = cv2.morphologyEx(
        gray,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_radius * 2 + 1, max_radius * 2 + 1)),
    )
    circles = cv2.HoughCircles(
        top_hat,
        cv2.HOUGH_GRADIENT,
        dp=1.15,
        minDist=max(18, min_radius * 3),
        param1=120,
        param2=10,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    results: List[DiscCircle] = []
    if circles is None:
        return results
    for x, y, radius in circles[0]:
        results.append((float(x), float(y), float(radius)))
    return results


def _detect_with_threshold(image: np.ndarray, gray: np.ndarray) -> List[DiscCircle]:
    original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    kernel_size = max(15, int(min(gray.shape[:2]) * 0.035))
    if kernel_size % 2 == 0:
        kernel_size += 1
    top_hat = cv2.morphologyEx(
        original_gray,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
    )
    threshold_value = max(12, int(np.percentile(top_hat, 85)))
    _, binary = cv2.threshold(top_hat, threshold_value, 255, cv2.THRESH_BINARY)
    strict_threshold = min(254, max(235, int(np.percentile(original_gray, 99.5))))
    _, strict_binary = cv2.threshold(original_gray, strict_threshold, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_or(binary, strict_binary)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    discs: List[DiscCircle] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 60:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        if circularity < 0.5:
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius > max(40, min(gray.shape[:2]) * 0.06):
            continue
        if not is_valid_disc(image, x, y, radius):
            continue
        discs.append((float(x), float(y), float(radius)))
    return discs


def detect_discs(image: np.ndarray) -> List[DiscCircle]:
    preprocessed = preprocess_plate(image)
    gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY) if preprocessed.ndim == 3 else preprocessed

    hough_discs = _detect_with_hough(gray)
    threshold_discs = _detect_with_threshold(image, gray)

    candidates = hough_discs + threshold_discs
    candidates = [disc for disc in candidates if is_valid_disc(image, *disc)]
    candidates = _merge_similar_discs(candidates)

    if not candidates:
        return []

    radii = np.array([disc[2] for disc in candidates], dtype=float)
    baseline_radius = float(np.percentile(radii, 30))
    filtered = [
        disc
        for disc in candidates
        if baseline_radius * 0.7 <= disc[2] <= baseline_radius * 1.45
    ]

    filtered = sorted(filtered, key=lambda disc: (round(disc[1] / 40), disc[0]))
    return filtered[: settings.MAX_DISCS]
