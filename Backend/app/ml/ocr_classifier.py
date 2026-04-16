from __future__ import annotations

import difflib
import re
from functools import lru_cache
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytesseract

ANTIBIOTIC_DICT = {
    "P", "AMP", "AM", "AMC", "SAM", "TZP", "PIP",
    "CAZ", "CTX", "CRO", "CEF", "CFM", "CPD", "CFX", "FOX",
    "IPM", "MEM", "ETP", "DOR",
    "ATM",
    "GEN", "AMK", "TOB", "KAN", "STR", "NET", "CN",
    "CIP", "NOR", "LEV", "OFX", "NAL", "GAT", "MOX",
    "TE", "DO", "TGC",
    "ERY", "AZM", "CLR",
    "VA", "TEC",
    "SXT", "TMP",
    "CHL",
    "CLI",
    "LZD",
    "RIF",
    "FOS", "MUP", "COL", "POL", "NIT", "FUR",
}
TEMPLATE_SIZE = (72, 144)


def _prepare_variants(roi_image: np.ndarray) -> List[np.ndarray]:
    resized = cv2.resize(roi_image, (192, 192))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
    gray = np.where(gray < 5, 255, gray).astype(np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    sharpened = cv2.addWeighted(enhanced, 1.5, cv2.GaussianBlur(enhanced, (0, 0), 2.0), -0.5, 0)
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)

    variants = [sharpened, binary, inverted]
    rotations: List[np.ndarray] = []
    for variant in variants:
        rotations.extend(
            [
                variant,
                cv2.rotate(variant, cv2.ROTATE_90_CLOCKWISE),
                cv2.rotate(variant, cv2.ROTATE_180),
                cv2.rotate(variant, cv2.ROTATE_90_COUNTERCLOCKWISE),
            ]
        )
    return rotations


def _normalize_text_mask(mask: np.ndarray) -> np.ndarray:
    working = mask.copy()
    if working.ndim == 3:
        working = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(working, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    text_mask = cv2.bitwise_not(binary)
    coords = cv2.findNonZero(text_mask)
    canvas = np.full(TEMPLATE_SIZE, 255, dtype=np.uint8)
    if coords is None:
        return canvas

    x, y, w, h = cv2.boundingRect(coords)
    cropped = text_mask[y : y + h, x : x + w]
    if w == 0 or h == 0:
        return canvas

    scale = min((TEMPLATE_SIZE[1] - 16) / w, (TEMPLATE_SIZE[0] - 16) / h)
    resized = cv2.resize(cropped, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)
    offset_y = (TEMPLATE_SIZE[0] - resized.shape[0]) // 2
    offset_x = (TEMPLATE_SIZE[1] - resized.shape[1]) // 2
    canvas[offset_y : offset_y + resized.shape[0], offset_x : offset_x + resized.shape[1]] = 255 - resized
    return canvas


@lru_cache(maxsize=128)
def _render_code_template(code: str) -> List[np.ndarray]:
    templates: List[np.ndarray] = []
    font_specs = [
        (cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2),
        (cv2.FONT_HERSHEY_DUPLEX, 1.15, 2),
        (cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.25, 1),
    ]
    for font, font_scale, thickness in font_specs:
        canvas = np.full(TEMPLATE_SIZE, 255, dtype=np.uint8)
        text_size, _ = cv2.getTextSize(code, font, font_scale, thickness)
        origin = (
            max(4, (TEMPLATE_SIZE[1] - text_size[0]) // 2),
            max(text_size[1] + 4, (TEMPLATE_SIZE[0] + text_size[1]) // 2),
        )
        cv2.putText(canvas, code, origin, font, font_scale, 0, thickness, cv2.LINE_AA)
        templates.append(canvas)
    return templates


def _predict_with_templates(roi_image: np.ndarray) -> Tuple[str, float]:
    candidate_mask = _normalize_text_mask(_prepare_variants(roi_image)[0])
    ink_fraction = float(np.mean(candidate_mask < 180))
    if ink_fraction < 0.005 or ink_fraction > 0.22:
        return "UNKNOWN", 0.0
    best_code = "UNKNOWN"
    best_score = 0.0
    second_best = 0.0
    for code in ANTIBIOTIC_DICT:
        for template in _render_code_template(code):
            diff = np.mean(np.abs(candidate_mask.astype(np.float32) - template.astype(np.float32))) / 255.0
            score = 1.0 - diff
            if score > best_score:
                second_best = best_score
                best_score = float(score)
                best_code = code
            elif score > second_best:
                second_best = float(score)
    return best_code, max(0.0, best_score - second_best)


def _score_token(token: str) -> Tuple[str, float]:
    cleaned = re.sub(r"[^A-Z]", "", token.upper())
    if not cleaned:
        return "UNKNOWN", 0.0

    exact = cleaned if cleaned in ANTIBIOTIC_DICT else None
    if exact:
        return exact, 0.98 if len(cleaned) >= 2 else 0.9

    match = difflib.get_close_matches(cleaned, ANTIBIOTIC_DICT, n=1, cutoff=0.5)
    if not match:
        return cleaned, 0.35

    ratio = difflib.SequenceMatcher(a=cleaned, b=match[0]).ratio()
    return match[0], float(ratio)


def predict_disc_class_ocr(roi_image: np.ndarray) -> Dict[str, object]:
    if roi_image is None or roi_image.size == 0:
        return {"code": "UNKNOWN", "confidence": 0.0, "candidates": []}

    best_code = "UNKNOWN"
    best_confidence = 0.0
    candidates: List[str] = []
    configs = [
        "--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    ]

    for variant in _prepare_variants(roi_image):
        for config in configs:
            text = pytesseract.image_to_string(variant, config=config)
            for token in re.findall(r"[A-Z0-9]{1,5}", text.upper()):
                code, score = _score_token(token)
                if code != "UNKNOWN" and code not in candidates:
                    candidates.append(code)
                if score > best_confidence:
                    best_code = code
                    best_confidence = score
        if best_confidence >= 0.97:
            break

    if best_code == "UNKNOWN" and candidates:
        best_code = candidates[0]

    if best_confidence < 0.72:
        template_code, template_score = _predict_with_templates(roi_image)
        if template_code not in candidates and template_code != "UNKNOWN":
            candidates.insert(0, template_code)
        template_confidence = round(max(0.0, min(0.74, template_score * 6.0)), 3)
        if template_confidence > best_confidence:
            best_code = template_code
            best_confidence = template_confidence

    return {
        "code": best_code,
        "confidence": round(best_confidence, 3),
        "candidates": candidates[:5],
    }
