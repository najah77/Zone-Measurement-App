from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from app.models.schemas import PlateDetection, Point


def _detect_plate_circle(image: np.ndarray, yolo_bbox: Optional[tuple] = None) -> Optional[Tuple[float, float, float, str]]:
    if yolo_bbox is not None:
        x1, y1, x2, y2 = yolo_bbox
        cx = float(x1 + x2) / 2.0
        cy = float(y1 + y2) / 2.0
        radius = float(max(x2 - x1, y2 - y1)) / 2.0
        return cx, cy, radius, "yolo-bbox"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.medianBlur(gray, 9)
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.dilate(edges, np.ones((5, 5), dtype=np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area > (image.shape[0] * image.shape[1] * 0.25):
            (x, y), radius = cv2.minEnclosingCircle(contour)
            return float(x), float(y), float(radius), "contour"

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.4,
        minDist=min(image.shape[:2]),
        param1=120,
        param2=28,
        minRadius=int(min(image.shape[:2]) * 0.25),
        maxRadius=int(min(image.shape[:2]) * 0.52),
    )
    if circles is not None:
        x, y, radius = circles[0][0]
        return float(x), float(y), float(radius), "hough"
    return None


def extract_plate(
    image: np.ndarray,
    yolo_bbox: Optional[tuple] = None,
) -> Tuple[np.ndarray, Optional[PlateDetection]]:
    if image is None or image.size == 0:
        return image, None

    detected = _detect_plate_circle(image, yolo_bbox=yolo_bbox)
    if detected is None:
        return image, None

    cx, cy, radius, method = detected
    h, w = image.shape[:2]
    margin = int(radius * 1.05)
    x1 = max(0, int(cx - margin))
    y1 = max(0, int(cy - margin))
    x2 = min(w, int(cx + margin))
    y2 = min(h, int(cy + margin))

    cropped = image[y1:y2, x1:x2].copy()
    clipped_pixels = 0
    if x1 == 0:
        clipped_pixels += int(max(0.0, margin - cx))
    if y1 == 0:
        clipped_pixels += int(max(0.0, margin - cy))
    if x2 == w:
        clipped_pixels += int(max(0.0, (cx + margin) - w))
    if y2 == h:
        clipped_pixels += int(max(0.0, (cy + margin) - h))

    clip_fraction = float(clipped_pixels) / max(1.0, radius * 4.0)
    warnings = []
    clipped = clip_fraction > 0.02
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
    return cropped if cropped.size else image, detection


def extract_plate_roi(image: np.ndarray, yolo_bbox: Optional[tuple] = None) -> np.ndarray:
    cropped, _ = extract_plate(image, yolo_bbox=yolo_bbox)
    return cropped
