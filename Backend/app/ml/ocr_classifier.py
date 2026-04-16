from __future__ import annotations

import difflib
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from app.utils.disc_features import extract_disc_features

ANTIBIOTIC_DICT = sorted(
    {
        "P",
        "AMP",
        "AM",
        "AMC",
        "SAM",
        "TZP",
        "PIP",
        "CAZ",
        "CTX",
        "CRO",
        "CEF",
        "CFM",
        "CPD",
        "CFX",
        "FOX",
        "IPM",
        "MEM",
        "ETP",
        "DOR",
        "ATM",
        "GEN",
        "AMK",
        "TOB",
        "KAN",
        "STR",
        "NET",
        "CN",
        "CIP",
        "NOR",
        "LEV",
        "OFX",
        "NAL",
        "GAT",
        "MOX",
        "TE",
        "DO",
        "TGC",
        "ERY",
        "AZM",
        "CLR",
        "VA",
        "TEC",
        "SXT",
        "TMP",
        "CHL",
        "CLI",
        "LZD",
        "RIF",
        "FOS",
        "MUP",
        "COL",
        "POL",
        "NIT",
        "FUR",
    }
)
TEMPLATE_SIZE = (96, 176)
ANGLE_STEPS = tuple(range(0, 360, 45))
CODE_STRENGTH_HINTS = {
    "P": {5},
    "SXT": {25},
    "FOX": {30},
    "TE": {30},
    "DO": {30},
    "LZD": {30},
}


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    if angle % 360 == 0:
        return image.copy()
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


def _tighten_disc_crop(roi_image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(roi_image, (240, 240), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
    gray = np.where(gray < 6, 255, gray).astype(np.uint8)

    bright_mask = cv2.inRange(gray, 180, 255)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return resized

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    pad = max(8, int(max(w, h) * 0.12))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(gray.shape[1], x + w + pad)
    y2 = min(gray.shape[0], y + h + pad)
    return resized[y1:y2, x1:x2].copy()


def _prepare_base_variants(roi_image: np.ndarray) -> List[np.ndarray]:
    crop = _tighten_disc_crop(roi_image)
    resized = cv2.resize(crop, (240, 240), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    sharpened = cv2.addWeighted(normalized, 1.8, cv2.GaussianBlur(normalized, (0, 0), 1.4), -0.8, 0)

    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        4,
    )
    inverted_adaptive = cv2.bitwise_not(adaptive)
    _, otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted_otsu = cv2.bitwise_not(otsu)
    top_hat = cv2.morphologyEx(
        sharpened,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
    )
    top_hat = cv2.normalize(top_hat, None, 0, 255, cv2.NORM_MINMAX)

    return [sharpened, inverted_adaptive, inverted_otsu, top_hat]


def _prepare_variants(roi_image: np.ndarray) -> List[np.ndarray]:
    variants: List[np.ndarray] = []
    for base in _prepare_base_variants(roi_image):
        for angle in ANGLE_STEPS:
            variants.append(_rotate_image(base, angle))
    return variants


def _normalize_text_mask(mask: np.ndarray) -> np.ndarray:
    working = mask.copy()
    if working.ndim == 3:
        working = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(working, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    text_mask = cv2.bitwise_not(binary)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    coords = cv2.findNonZero(text_mask)
    canvas = np.full(TEMPLATE_SIZE, 255, dtype=np.uint8)
    if coords is None:
        return canvas

    x, y, w, h = cv2.boundingRect(coords)
    if w == 0 or h == 0:
        return canvas
    cropped = text_mask[y : y + h, x : x + w]
    scale = min((TEMPLATE_SIZE[1] - 20) / w, (TEMPLATE_SIZE[0] - 20) / h)
    resized = cv2.resize(
        cropped,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_LINEAR,
    )
    offset_y = (TEMPLATE_SIZE[0] - resized.shape[0]) // 2
    offset_x = (TEMPLATE_SIZE[1] - resized.shape[1]) // 2
    canvas[offset_y : offset_y + resized.shape[0], offset_x : offset_x + resized.shape[1]] = 255 - resized
    return canvas


@lru_cache(maxsize=128)
def _render_code_template(code: str) -> List[np.ndarray]:
    templates: List[np.ndarray] = []
    font_specs = [
        (cv2.FONT_HERSHEY_SIMPLEX, 1.25, 2),
        (cv2.FONT_HERSHEY_DUPLEX, 1.2, 2),
        (cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3, 2),
    ]
    for font, font_scale, thickness in font_specs:
        canvas = np.full(TEMPLATE_SIZE, 255, dtype=np.uint8)
        text_size, _ = cv2.getTextSize(code, font, font_scale, thickness)
        origin = (
            max(4, (TEMPLATE_SIZE[1] - text_size[0]) // 2),
            max(text_size[1] + 6, (TEMPLATE_SIZE[0] + text_size[1]) // 2),
        )
        cv2.putText(canvas, code, origin, font, font_scale, 0, thickness, cv2.LINE_AA)
        templates.append(canvas)
    return templates


def _template_score(mask: np.ndarray, strength: int | None) -> Tuple[str, float]:
    ink_fraction = float(np.mean(mask < 180))
    if ink_fraction < 0.003 or ink_fraction > 0.32:
        return "UNKNOWN", 0.0

    best_code = "UNKNOWN"
    best_score = 0.0
    second_best = 0.0
    for code in ANTIBIOTIC_DICT:
        strength_bonus = 0.035 if strength is not None and strength in CODE_STRENGTH_HINTS.get(code, set()) else 0.0
        for template in _render_code_template(code):
            diff = np.mean(np.abs(mask.astype(np.float32) - template.astype(np.float32))) / 255.0
            score = 1.0 - diff + strength_bonus
            if score > best_score:
                second_best = best_score
                best_score = float(score)
                best_code = code
            elif score > second_best:
                second_best = float(score)
    return best_code, max(0.0, best_score - second_best)


def _score_candidate(code: str, cleaned: str, strength: int | None, conf_hint: float) -> float:
    if code == "UNKNOWN":
        return 0.0
    if cleaned == code:
        base = 0.84
    else:
        ratio = difflib.SequenceMatcher(a=cleaned, b=code).ratio()
        if cleaned and (cleaned in code or code in cleaned):
            base = 0.72 + min(0.12, ratio * 0.12)
        elif ratio >= 0.45:
            base = 0.42 + ratio * 0.38
        else:
            return 0.0
    if strength is not None and strength in CODE_STRENGTH_HINTS.get(code, set()):
        base += 0.08
    base += min(0.12, conf_hint * 0.0015)
    return min(0.99, base)


def _score_token(token: str, strength: int | None, conf_hint: float = 0.0) -> Tuple[str, float]:
    cleaned = re.sub(r"[^A-Z]", "", token.upper())
    if not cleaned:
        return "UNKNOWN", 0.0

    best_code = "UNKNOWN"
    best_score = 0.0
    for code in ANTIBIOTIC_DICT:
        score = _score_candidate(code, cleaned, strength, conf_hint)
        if score > best_score:
            best_code = code
            best_score = score
    return best_code, round(best_score, 3)


def _extract_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Z0-9]{1,6}", text.upper())


def _collect_ocr_candidates(variant: np.ndarray, strength: int | None) -> List[Tuple[str, float]]:
    candidates: List[Tuple[str, float]] = []
    configs = [
        "--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    ]
    for config in configs:
        data = pytesseract.image_to_data(variant, config=config, output_type=Output.DICT)
        texts = data.get("text", [])
        confidences = data.get("conf", [])
        for raw_text, raw_conf in zip(texts, confidences):
            try:
                conf_value = float(raw_conf)
            except (TypeError, ValueError):
                conf_value = 0.0
            for token in _extract_tokens(str(raw_text)):
                code, score = _score_token(token, strength, conf_value)
                if code != "UNKNOWN":
                    candidates.append((code, score))
    return candidates


def _merge_scores(pairs: Iterable[Tuple[str, float]]) -> List[Tuple[str, float]]:
    merged: Dict[str, float] = {}
    for code, score in pairs:
        merged[code] = max(score, merged.get(code, 0.0))
    return sorted(merged.items(), key=lambda item: item[1], reverse=True)


def predict_disc_class_ocr(roi_image: np.ndarray) -> Dict[str, object]:
    if roi_image is None or roi_image.size == 0:
        return {"code": "UNKNOWN", "confidence": 0.0, "candidates": []}

    features = extract_disc_features(roi_image)
    strength = features.get("strength")
    scored_candidates: List[Tuple[str, float]] = []
    template_candidates: List[Tuple[str, float]] = []

    for variant in _prepare_variants(roi_image):
        scored_candidates.extend(_collect_ocr_candidates(variant, strength))
        mask = _normalize_text_mask(variant)
        template_code, template_margin = _template_score(mask, strength)
        if template_code != "UNKNOWN":
            template_candidates.append((template_code, min(0.9, 0.46 + template_margin * 4.2)))

    merged = _merge_scores(scored_candidates + template_candidates)
    if not merged:
        return {"code": "UNKNOWN", "confidence": 0.0, "candidates": []}

    best_code, best_score = merged[0]
    second_score = merged[1][1] if len(merged) > 1 else 0.0
    strength_bonus = 0.04 if strength is not None and strength in CODE_STRENGTH_HINTS.get(best_code, set()) else 0.0
    confidence = min(0.99, best_score + strength_bonus + max(0.0, (best_score - second_score) * 0.12))

    candidates = [code for code, _ in merged[:5]]
    return {
        "code": best_code,
        "confidence": round(confidence, 3),
        "candidates": candidates,
        "strength": strength,
        "raw_ocr": features.get("raw_ocr", ""),
    }
