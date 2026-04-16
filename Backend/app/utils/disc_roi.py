from __future__ import annotations

import cv2
import numpy as np


def extract_disc_roi(
    image: np.ndarray,
    x: float,
    y: float,
    r: float,
    expand_ratio: float = 0.32,
    mask_scale: float = 0.82,
) -> np.ndarray:
    """
    Crop a disc-focused ROI for OCR/classification.

    The detected disc radius can include some halo pixels on real plates, so the
    OCR crop is intentionally tighter than the measurement crop.
    """
    if image is None or image.size == 0 or r <= 0:
        return np.array([])

    img_h, img_w = image.shape[:2]
    crop_radius = int(round(r * (1.0 + expand_ratio)))
    if crop_radius <= 0:
        return np.array([])

    cx = int(round(x))
    cy = int(round(y))
    x1 = max(0, cx - crop_radius)
    y1 = max(0, cy - crop_radius)
    x2 = min(img_w, cx + crop_radius)
    y2 = min(img_h, cy + crop_radius)
    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return np.array([])

    h, w = crop.shape[:2]
    center = (w // 2, h // 2)
    mask_radius = max(4, int(round(r * mask_scale)))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, mask_radius, 255, -1)

    if crop.ndim == 3:
        masked = crop.copy()
        masked[mask == 0] = 255
    else:
        masked = crop.copy()
        masked[mask == 0] = 255
    return masked
