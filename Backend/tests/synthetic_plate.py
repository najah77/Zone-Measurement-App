from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class DiscSpec:
    center: Tuple[int, int]
    zone_radius_px: int
    code: str = ""


def encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode synthetic fixture.")
    return encoded.tobytes()


def make_plate_image(
    specs: Sequence[DiscSpec],
    image_size: int = 820,
    plate_radius: int = 340,
    disc_radius: int = 22,
    render_labels: bool = False,
) -> np.ndarray:
    image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 175
    center = image_size // 2
    cv2.circle(image, (center, center), plate_radius, (205, 205, 205), -1)

    for spec in specs:
        if spec.zone_radius_px > disc_radius:
            cv2.circle(image, spec.center, spec.zone_radius_px, (232, 232, 232), -1)
        cv2.circle(image, spec.center, disc_radius, (255, 255, 255), -1)
        if render_labels and spec.code:
            text_size, _ = cv2.getTextSize(spec.code, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            origin = (
                spec.center[0] - text_size[0] // 2,
                spec.center[1] + text_size[1] // 2,
            )
            cv2.putText(
                image,
                spec.code,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (10, 10, 10),
                1,
                cv2.LINE_AA,
            )
    return image


def make_label_crop(code: str, size: int = 180) -> np.ndarray:
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    cv2.circle(image, (size // 2, size // 2), size // 3, (255, 255, 255), -1)
    text_size, _ = cv2.getTextSize(code, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    origin = ((size - text_size[0]) // 2, (size + text_size[1]) // 2)
    cv2.putText(image, code, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return image
