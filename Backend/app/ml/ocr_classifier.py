from __future__ import annotations

import re
from itertools import zip_longest
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import pytesseract

from app.core.config import settings
from app.utils.preprocessing import apply_clahe

ANTIBIOTIC_CODEBOOK: Dict[str, Dict[str, object]] = {
    "P": {"strengths": {10}},
    "AMP": {"strengths": {10}},
    "AM": {"strengths": {10}},
    "AM10": {"strengths": {10}},
    "AMC": {"strengths": {30}},
    "SAM": {"strengths": {20}},
    "TZP": {"strengths": {110}},
    "TPZ": {"strengths": {36, 110}},
    "PIP": {"strengths": {100}},
    "CAZ": {"strengths": {30}},
    "CTX": {"strengths": {30}},
    "CRO": {"strengths": {30}},
    "CEF": {"strengths": {30}},
    "CFM": {"strengths": {5}},
    "CPD": {"strengths": {10}},
    "CFX": {"strengths": {30}},
    "FOX": {"strengths": {30}},
    "FEP": {"strengths": {30}},
    "ATM": {"strengths": {30}},
    "IPM": {"strengths": {10}},
    "MEM": {"strengths": {10}},
    "ETP": {"strengths": {10}},
    "DOR": {"strengths": {10}},
    "GEN": {"strengths": {10}},
    "CN": {"strengths": {10}},
    "AMK": {"strengths": {30}},
    "AK": {"strengths": {30}},
    "TOB": {"strengths": {10}},
    "TOP": {"strengths": {10}},
    "KAN": {"strengths": {30}},
    "STR": {"strengths": {10}},
    "NET": {"strengths": {30}},
    "CIP": {"strengths": {5}},
    "PEF": {"strengths": {5}},
    "NOR": {"strengths": {10}},
    "LEV": {"strengths": {5}},
    "OFX": {"strengths": {5}},
    "NAL": {"strengths": {30}},
    "GAT": {"strengths": {5}},
    "MOX": {"strengths": {5}},
    "TE": {"strengths": {30}},
    "DO": {"strengths": {30}},
    "TGC": {"strengths": {15}},
    "MIN": {"strengths": {30}},
    "ERY": {"strengths": {15}},
    "AZM": {"strengths": {15}},
    "CLR": {"strengths": {15}},
    "VA": {"strengths": {30}},
    "TEC": {"strengths": {30}},
    "SXT": {"strengths": {25}},
    "TMP": {"strengths": {5}},
    "CHL": {"strengths": {30}},
    "CLI": {"strengths": {2}},
    "LZD": {"strengths": {30}},
    "RIF": {"strengths": {5}},
    "FOS": {"strengths": {50, 200}},
    "FF": {"strengths": {200}},
    "MUP": {"strengths": {5, 200}},
    "COL": {"strengths": {10}},
    "POL": {"strengths": {300}},
    "NIT": {"strengths": {300}},
    "FUR": {"strengths": {100}},
    "F100": {"strengths": {100}},
}

STRONG_ALIAS_MAP = {
    "GN": "CN",
    "CIF": "CIP",
    "TOP": "TOB",
}
SOFT_ALIAS_MAP = {
    "IE": "TE",
    "SAN": "SAM",
}

VALID_STRENGTHS = {2, 5, 10, 15, 20, 25, 30, 50, 100, 110, 200, 300}
PSM_CONFIGS: Sequence[Tuple[int, str]] = (
    (6, "--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    (11, "--psm 11 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
)
POLAR_PSM_CONFIG: Tuple[int, str] = (
    7,
    "--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
)
VARIANT_WEIGHTS = {
    "sharpened": 1.0,
    "blackhat": 1.08,
    "inv_otsu": 1.04,
    "dilate": 1.02,
    "adaptive": 0.98,
    "polar_strip": 1.1,
    "polar_blackhat": 1.14,
}
PSM_WEIGHTS = {6: 1.0, 11: 0.97}
ANGLE_OFFSETS = (-12, 0, 12)
FALLBACK_ANGLES = (0, 45, 90, 135, 180, 225, 270, 315)
PRIMARY_VARIANTS = ("blackhat", "inv_otsu")
SECONDARY_VARIANTS = ("adaptive", "sharpened")
OCR_CHAR_TRANSLATION = str.maketrans(
    {
        "0": "O",
        "1": "I",
        "2": "Z",
        "4": "A",
        "5": "S",
        "6": "G",
        "7": "T",
        "8": "B",
    }
)
STRENGTH_CHAR_TRANSLATION = str.maketrans(
    {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "T": "1",
        "S": "5",
        "B": "8",
    }
)
CONFUSABLE_CHAR_GROUPS = (
    {"O", "Q", "D", "0"},
    {"I", "L", "1", "T"},
    {"S", "5"},
    {"Z", "2"},
    {"B", "8"},
    {"M", "N"},
    {"C", "G"},
    {"V", "Y"},
    {"P", "F"},
)


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    if angle % 360 == 0:
        return image.copy()
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


def _tighten_disc_crop(roi_image: np.ndarray) -> np.ndarray:
    if roi_image.ndim == 3:
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_image.copy()
    resized = cv2.resize(gray, (320, 320), interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(resized, None, 7, 7, 21)

    bright_mask = cv2.inRange(denoised, 172, 255)
    bright_mask = cv2.morphologyEx(
        bright_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return denoised

    contour = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    if radius <= 4:
        return denoised

    padded_radius = min(radius * 1.12, min(denoised.shape[:2]) / 2.0 - 4.0)
    x1 = max(0, int(cx - padded_radius))
    y1 = max(0, int(cy - padded_radius))
    x2 = min(denoised.shape[1], int(cx + padded_radius))
    y2 = min(denoised.shape[0], int(cy + padded_radius))
    cropped = denoised[y1:y2, x1:x2].copy()
    if cropped.size == 0:
        return denoised

    h, w = cropped.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    circle_center = (int(cx - x1), int(cy - y1))
    circle_radius = max(6, int(round(min(padded_radius * 0.98, min(h, w) / 2.0 - 2.0))))
    cv2.circle(mask, circle_center, circle_radius, 255, -1)
    cropped[mask == 0] = 255
    return cv2.resize(cropped, (320, 320), interpolation=cv2.INTER_CUBIC)


def _build_variants(roi_image: np.ndarray) -> Dict[str, np.ndarray]:
    crop = _tighten_disc_crop(roi_image)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(crop)
    blurred = cv2.GaussianBlur(clahe, (3, 3), 0)
    sharpened = cv2.addWeighted(clahe, 1.65, cv2.GaussianBlur(clahe, (0, 0), 1.1), -0.65, 0)

    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, blackhat_kernel)
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)

    _, otsu = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv_otsu = cv2.bitwise_not(otsu)
    dilate = cv2.dilate(inv_otsu, np.ones((3, 3), dtype=np.uint8), iterations=1)
    adaptive = cv2.adaptiveThreshold(
        blackhat,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        4,
    )

    return {
        "sharpened": sharpened,
        "blackhat": blackhat,
        "inv_otsu": inv_otsu,
        "dilate": dilate,
        "adaptive": adaptive,
    }


def _build_polar_variants(roi_image: np.ndarray) -> Dict[str, np.ndarray]:
    crop = _tighten_disc_crop(roi_image)
    height, width = crop.shape[:2]
    center = (width / 2.0, height / 2.0)
    max_radius = max(8.0, min(height, width) / 2.0 - 2.0)
    polar = cv2.warpPolar(
        crop,
        (int(round(max_radius)), 360),
        center,
        max_radius,
        cv2.WARP_POLAR_LINEAR,
    )
    inner_radius = int(round(max_radius * 0.42))
    outer_radius = int(round(max_radius * 0.96))
    annulus = polar[:, inner_radius:outer_radius]
    if annulus.size == 0:
        return {}

    strip = cv2.transpose(annulus)
    strip = cv2.resize(strip, (720, 140), interpolation=cv2.INTER_CUBIC)
    strip = apply_clahe(strip, clip_limit=2.6)
    blackhat = cv2.morphologyEx(
        strip,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (31, 9)),
    )
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    _, thresholded = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strip_variants = {
        "polar_strip": strip,
        "polar_blackhat": cv2.bitwise_not(thresholded),
    }
    return strip_variants


def build_ocr_debug_artifacts(roi_image: np.ndarray) -> Dict[str, np.ndarray]:
    if roi_image is None or roi_image.size == 0:
        return {}
    tightened = _tighten_disc_crop(roi_image)
    variants = _build_variants(roi_image)
    polar_variants = _build_polar_variants(roi_image)
    artifacts: Dict[str, np.ndarray] = {
        "tightened_crop": tightened,
        "gray": tightened,
    }
    artifacts.update(variants)
    artifacts.update(polar_variants)
    return artifacts


def _estimate_candidate_angles(blackhat: np.ndarray) -> List[int]:
    percentile = int(np.percentile(blackhat, 97))
    mask = cv2.inRange(blackhat, percentile, 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    coords = cv2.findNonZero(mask)
    if coords is None or len(coords) < 8:
        return list(FALLBACK_ANGLES)

    rect = cv2.minAreaRect(coords)
    base_angle = float(rect[-1]) % 360.0
    prioritized: List[int] = []
    seen = set()

    def _push(angle_value: float) -> None:
        normalized = int(round(angle_value % 360.0))
        if normalized not in seen:
            seen.add(normalized)
            prioritized.append(normalized)

    for base in (base_angle, base_angle + 90.0, base_angle + 180.0, base_angle + 270.0):
        _push(base)
    for base in (base_angle, base_angle + 180.0):
        for offset in (-10.0, 10.0):
            _push(base + offset)

    return prioritized[:8] if prioritized else list(FALLBACK_ANGLES)


def _extract_code_tokens(raw_text: str) -> List[str]:
    tokens = re.findall(r"[A-Z0-9]{1,6}", raw_text.upper())
    code_tokens: List[str] = []
    seen = set()
    for token in tokens:
        if token.isdigit():
            continue
        alpha_candidate = token.translate(OCR_CHAR_TRANSLATION)
        alpha_candidate = re.sub(r"[^A-Z]", "", alpha_candidate)
        if alpha_candidate:
            alpha_candidate = alpha_candidate[:5]
            if alpha_candidate not in seen:
                seen.add(alpha_candidate)
                code_tokens.append(alpha_candidate)

        alnum_token = re.sub(r"[^A-Z0-9]", "", token.upper())
        if alnum_token:
            for split_index in range(1, min(5, len(alnum_token))):
                prefix_text = alnum_token[:split_index]
                suffix_text = alnum_token[split_index:]
                letter_prefix = re.sub(r"[^A-Z]", "", prefix_text.translate(OCR_CHAR_TRANSLATION))
                suffix_digits = suffix_text.translate(STRENGTH_CHAR_TRANSLATION)
                suffix_digits = re.sub(r"[^0-9]", "", suffix_digits)
                if letter_prefix and suffix_digits:
                    alnum_candidate = f"{letter_prefix[:4]}{suffix_digits[:3]}"
                    if alnum_candidate not in seen:
                        seen.add(alnum_candidate)
                        code_tokens.append(alnum_candidate)
    return code_tokens


def _extract_strengths(raw_text: str) -> List[int]:
    strengths: List[int] = []
    seen = set()
    tokens = re.findall(r"[A-Z0-9]{1,6}", raw_text.upper())
    for token in tokens:
        has_digits = any(character.isdigit() for character in token)
        if not has_digits:
            if len(token) < 2:
                continue
            if not all(character in {"O", "Q", "D", "I", "L", "T", "S", "B"} for character in token):
                continue
        normalized = token.translate(STRENGTH_CHAR_TRANSLATION)
        for digits in re.findall(r"\d{1,3}", normalized):
            value = int(digits)
            if value in VALID_STRENGTHS and value not in seen:
                seen.add(value)
                strengths.append(value)
    return strengths


def _is_confusable(left: str, right: str) -> bool:
    return any(left in group and right in group for group in CONFUSABLE_CHAR_GROUPS)


def _is_ordered_subsequence(token: str, candidate: str) -> bool:
    token_index = 0
    candidate_index = 0
    while token_index < len(token) and candidate_index < len(candidate):
        if token[token_index] == candidate[candidate_index]:
            token_index += 1
        candidate_index += 1
    return token_index == len(token)


def _alignment_score(token: str, candidate: str) -> Tuple[float, str]:
    alias_target = STRONG_ALIAS_MAP.get(token)
    soft_alias_target = SOFT_ALIAS_MAP.get(token)
    if token == candidate:
        return 1.0, "exact"
    if alias_target == candidate:
        return 0.95, "alias"
    if soft_alias_target == candidate:
        return 0.82, "soft_alias"
    if len(token) == 1 and len(candidate) > 1:
        return 0.0, "too_short"
    if len(token) == 2 and len(candidate) == 3:
        if token == candidate[:2]:
            return 0.9, "prefix"
        if token == candidate[1:]:
            return 0.84, "suffix"
        if token == (candidate[0] + candidate[2]):
            return 0.88, "outer-pair"
    if len(token) == 3 and len(candidate) == 4:
        if token == candidate[:3]:
            return 0.9, "prefix"
        if token == candidate[1:]:
            return 0.84, "suffix"
        if token == (candidate[0] + candidate[2] + candidate[3]):
            return 0.86, "skip-middle"
    if abs(len(token) - len(candidate)) > 1:
        return 0.0, "length_mismatch"

    max_len = max(len(token), len(candidate))
    penalty = 0.0
    for left, right in zip_longest(token, candidate, fillvalue=""):
        if not left or not right:
            penalty += 0.38
        elif left == right:
            continue
        elif _is_confusable(left, right):
            penalty += 0.22
        else:
            penalty += 0.48

    score = max(0.0, 1.0 - (penalty / float(max_len)))
    coverage = min(len(token), len(candidate)) / float(max_len)
    score *= 0.55 + (coverage * 0.45)
    if _is_ordered_subsequence(token, candidate):
        score += 0.08
    elif len(token) >= 2:
        score -= 0.12
    if len(token) == 2 and len(candidate) == 3 and token in {candidate[:2], candidate[1:]}:
        score += 0.08
    elif len(token) == 2 and len(candidate) == 3 and token == (candidate[0] + candidate[2]):
        score += 0.04
    return min(0.96, score), "partial"


def _strength_adjustment(candidate: str, strengths: Sequence[int]) -> Tuple[float, str]:
    if not strengths:
        return 0.0, ""
    expected_strengths = set(ANTIBIOTIC_CODEBOOK.get(candidate, {}).get("strengths", set()))
    if not expected_strengths:
        return 0.0, ""
    if any(strength in expected_strengths for strength in strengths):
        return 0.18, f"strength {'/'.join(str(value) for value in strengths)}"
    return -0.18, f"strength mismatch {'/'.join(str(value) for value in strengths)}"


def _normalize_observation_text(tokens: Sequence[str], strengths: Sequence[int]) -> str:
    parts: List[str] = []
    if tokens:
        parts.append(tokens[0])
    if strengths:
        parts.append("/".join(strengths[i].__str__() for i in range(min(len(strengths), 2))))
    return " ".join(parts).strip()


def _is_rank_confident_enough(ranked: Sequence[Dict[str, object]]) -> bool:
    if not ranked:
        return False
    top = ranked[0]
    top_score = float(top.get("score", 0.0))
    second_score = float(ranked[1]["score"]) if len(ranked) > 1 else 0.0
    margin = top_score - second_score
    return (
        int(top.get("exact_matches", 0)) > 0
        and (
            top_score >= settings.OCR_HIGH_CONFIDENCE
            or (top_score >= settings.OCR_MIN_CONFIDENCE and margin >= (settings.OCR_MIN_MARGIN * 0.6))
        )
    )


def _collect_ocr_pass(
    observations: List[Dict[str, object]],
    variants: Dict[str, np.ndarray],
    variant_names: Sequence[str],
    angles: Sequence[int],
    psm_configs: Sequence[Tuple[int, str]],
) -> None:
    for variant_name in variant_names:
        variant_image = variants[variant_name]
        for angle in angles:
            rotated = _rotate_image(variant_image, angle)
            for psm, config in psm_configs:
                try:
                    raw_text = pytesseract.image_to_string(rotated, config=config)
                except pytesseract.TesseractError:
                    continue
                raw_text = raw_text.strip()
                if not raw_text:
                    continue

                code_tokens = _extract_code_tokens(raw_text)
                strengths = _extract_strengths(raw_text)
                if not code_tokens and not strengths:
                    continue

                observations.append(
                    {
                        "variant": variant_name,
                        "angle": angle,
                        "psm": psm,
                        "raw_text": raw_text.replace("\n", " | "),
                        "tokens": code_tokens,
                        "strengths": strengths,
                        "normalized_text": _normalize_observation_text(code_tokens, strengths),
                    }
                )


def _ocr_observations(roi_image: np.ndarray) -> List[Dict[str, object]]:
    variants = _build_variants(roi_image)
    candidate_angles = _estimate_candidate_angles(variants["blackhat"])
    primary_angles = candidate_angles[:4] if candidate_angles else list(FALLBACK_ANGLES[:4])
    fallback_angles = candidate_angles[4:] if len(candidate_angles) > 4 else list(FALLBACK_ANGLES[4:])
    observations: List[Dict[str, object]] = []

    _collect_ocr_pass(observations, variants, PRIMARY_VARIANTS, primary_angles, (PSM_CONFIGS[0],))
    ranked = _score_candidates(observations)
    if _is_rank_confident_enough(ranked):
        return observations

    _collect_ocr_pass(observations, variants, PRIMARY_VARIANTS, primary_angles, (PSM_CONFIGS[1],))
    ranked = _score_candidates(observations)
    if _is_rank_confident_enough(ranked):
        return observations

    polar_variants = _build_polar_variants(roi_image)
    if polar_variants:
        _collect_ocr_pass(
            observations,
            polar_variants,
            tuple(polar_variants.keys()),
            (0, 90, 180, 270),
            (POLAR_PSM_CONFIG,),
        )
        ranked = _score_candidates(observations)
        if _is_rank_confident_enough(ranked):
            return observations

    secondary_angles = primary_angles + fallback_angles[:4]
    _collect_ocr_pass(observations, variants, SECONDARY_VARIANTS, secondary_angles, PSM_CONFIGS)
    return observations


def _score_candidates(observations: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    observed_strength_counter: Dict[int, int] = {}
    for observation in observations:
        for strength in observation.get("strengths", []):
            try:
                normalized_strength = int(strength)
            except (TypeError, ValueError):
                continue
            observed_strength_counter[normalized_strength] = observed_strength_counter.get(normalized_strength, 0) + 1

    aggregate: Dict[str, Dict[str, object]] = {
        code: {
            "code": code,
            "scores": [],
            "exact_matches": 0,
            "alias_matches": 0,
            "strength_hits": 0,
            "global_strength_matches": 0,
            "structured_matches": 0,
            "angles": set(),
            "sources": set(),
            "tokens": set(),
            "best_observation": None,
            "best_score": 0.0,
            "reasons": [],
        }
        for code in ANTIBIOTIC_CODEBOOK
    }

    for observation in observations:
        tokens = list(observation.get("tokens", []))
        strengths = list(observation.get("strengths", []))
        if not tokens:
            continue

        scored_for_observation: List[Tuple[str, float, str, str, float]] = []
        for token in tokens:
            for candidate in ANTIBIOTIC_CODEBOOK:
                alignment_score, match_type = _alignment_score(token, candidate)
                if alignment_score <= 0.0:
                    continue
                strength_delta, strength_reason = _strength_adjustment(candidate, strengths)
                total = alignment_score
                total *= VARIANT_WEIGHTS.get(str(observation["variant"]), 1.0)
                total *= PSM_WEIGHTS.get(int(observation["psm"]), 1.0)
                total += strength_delta
                if match_type == "exact":
                    total += 0.06
                elif match_type == "alias":
                    total += 0.04
                total = max(0.0, min(1.0, total))
                if total < 0.40:
                    continue
                reason_bits = [match_type]
                if strength_reason:
                    reason_bits.append(strength_reason)
                scored_for_observation.append((candidate, total, token, ", ".join(reason_bits), strength_delta))

        scored_for_observation.sort(key=lambda item: item[1], reverse=True)
        for candidate, total, token, reason, strength_delta in scored_for_observation[:6]:
            entry = aggregate[candidate]
            entry["scores"].append(total)
            entry["angles"].add(int(observation["angle"]))
            entry["sources"].add(int(observation.get("source_index", 0)))
            entry["tokens"].add(token)
            entry["reasons"].append(reason)
            if strength_delta > 0:
                entry["strength_hits"] += 1
            if token == candidate:
                entry["exact_matches"] += 1
            elif STRONG_ALIAS_MAP.get(token) == candidate:
                entry["alias_matches"] += 1
            if reason.startswith(("prefix", "suffix", "outer-pair", "skip-middle")):
                entry["structured_matches"] += 1
            if total > float(entry["best_score"]):
                entry["best_score"] = total
                entry["best_observation"] = {
                    "raw_text": observation["raw_text"],
                    "normalized_text": observation["normalized_text"],
                    "angle": observation["angle"],
                    "variant": observation["variant"],
                    "source_index": observation.get("source_index", 0),
                    "token": token,
                    "reason": reason,
                    "strengths": strengths,
                }

    ranked: List[Dict[str, object]] = []
    for entry in aggregate.values():
        if not entry["scores"]:
            continue
        scores = sorted((float(value) for value in entry["scores"]), reverse=True)
        best_score = scores[0]
        top_slice = scores[: min(3, len(scores))]
        mean_top = sum(top_slice) / float(len(top_slice))
        support_bonus = min(0.10, 0.035 * max(0, len(scores) - 1))
        angle_bonus = min(0.04, 0.015 * max(0, len(entry["angles"]) - 1))
        source_bonus = min(0.04, 0.02 * max(0, len(entry["sources"]) - 1))
        combined_score = min(0.99, (best_score * 0.72) + (mean_top * 0.28) + support_bonus + angle_bonus + source_bonus)
        expected_strengths = set(ANTIBIOTIC_CODEBOOK.get(str(entry["code"]), {}).get("strengths", set()))
        global_strength_matches = sum(observed_strength_counter.get(value, 0) for value in expected_strengths)
        entry["global_strength_matches"] = global_strength_matches
        if global_strength_matches > 0 and len(scores) > 0:
            combined_score = min(
                0.99,
                combined_score + min(0.09, 0.025 * global_strength_matches + (0.02 if int(entry["structured_matches"]) > 0 else 0.0)),
            )
        if int(entry["structured_matches"]) > 1 and len(str(entry["code"])) >= 3:
            combined_score = min(0.99, combined_score + min(0.06, 0.02 * int(entry["structured_matches"])))
        if int(entry["exact_matches"]) == 0:
            combined_score = min(combined_score, 0.94)
        if int(entry["exact_matches"]) == 0 and int(entry["alias_matches"]) > 0:
            combined_score = min(combined_score, 0.88)
        if len(scores) == 1 and int(entry["exact_matches"]) == 0:
            combined_score = max(0.0, combined_score - 0.06)
        if len(str(entry["code"])) == 1 and int(entry["exact_matches"]) < 2:
            combined_score = min(combined_score, 0.84)
        if len(str(entry["code"])) <= 2 and int(entry["exact_matches"]) == 0 and int(entry["strength_hits"]) == 0:
            combined_score = min(combined_score, 0.86)
        if str(entry["code"]) == "AM" and int(entry["exact_matches"]) == 0 and global_strength_matches == 0:
            combined_score = min(combined_score, 0.74)
        ranked.append(
            {
                "code": entry["code"],
                "score": round(combined_score, 3),
                "exact_matches": int(entry["exact_matches"]),
                "alias_matches": int(entry["alias_matches"]),
                "strength_hits": int(entry["strength_hits"]),
                "global_strength_matches": int(global_strength_matches),
                "structured_matches": int(entry["structured_matches"]),
                "support_count": len(scores),
                "source_count": len(entry["sources"]),
                "best_strength_count": len(list((entry["best_observation"] or {}).get("strengths", []))),
                "best_observation": entry["best_observation"],
                "token_support": sorted(str(value) for value in entry["tokens"])[:6],
            }
        )

    ranked.sort(
        key=lambda item: (
            -float(item["score"]),
            -int(item["strength_hits"]),
            -int(item.get("global_strength_matches", 0)),
            -int(item.get("structured_matches", 0)),
            -int(item.get("best_strength_count", 0)),
            -int(item["exact_matches"]),
            -int(item["support_count"]),
            -len(str(item["code"])),
            str(item["code"]),
        )
    )
    if ranked and str(ranked[0]["code"]) == "P" and int(ranked[0]["exact_matches"]) == 0:
        for alternative in ranked[1:6]:
            code = str(alternative["code"])
            observation = dict(alternative.get("best_observation") or {})
            token = str(observation.get("token", ""))
            if len(code) < 3 or not token:
                continue
            if float(alternative["score"]) < (float(ranked[0]["score"]) - 0.12):
                continue
            structured_match = token == code or code.startswith(token) or code.endswith(token)
            if len(token) == 2 and len(code) == 3:
                structured_match = structured_match or token == (code[0] + code[2])
            if not structured_match:
                continue
            if int(alternative["strength_hits"]) == 0 and int(alternative["exact_matches"]) == 0:
                continue
            ranked.insert(0, ranked.pop(ranked.index(alternative)))
            break
    if (
        ranked
        and len(str(ranked[0]["code"])) <= 2
        and int(ranked[0]["strength_hits"]) == 0
        and int(ranked[0]["exact_matches"]) == 0
    ):
        for alternative in ranked[1:4]:
            if len(str(alternative["code"])) < 3:
                continue
            if float(alternative["score"]) < (float(ranked[0]["score"]) - 0.12):
                continue
            if int(alternative["strength_hits"]) == 0 and int(alternative["exact_matches"]) == 0:
                continue
            ranked.insert(0, ranked.pop(ranked.index(alternative)))
            break
    return ranked


def _build_ocr_debug_payload(
    observations: Sequence[Dict[str, object]],
    ranked: Sequence[Dict[str, object]],
) -> Dict[str, Any]:
    return {
        "observation_count": len(observations),
        "observations": [
            {
                "source_index": int(observation.get("source_index", 0)),
                "variant": str(observation.get("variant", "")),
                "angle": int(observation.get("angle", 0)),
                "psm": int(observation.get("psm", 0)),
                "raw_text": str(observation.get("raw_text", "")),
                "normalized_text": str(observation.get("normalized_text", "")),
                "tokens": list(observation.get("tokens", []))[:6],
                "strengths": [int(value) for value in observation.get("strengths", [])[:4]],
            }
            for observation in observations[:40]
        ],
        "candidate_ranking": [
            {
                "code": str(item.get("code", "")),
                "score": float(item.get("score", 0.0)),
                "exact_matches": int(item.get("exact_matches", 0)),
                "alias_matches": int(item.get("alias_matches", 0)),
                "strength_hits": int(item.get("strength_hits", 0)),
                "global_strength_matches": int(item.get("global_strength_matches", 0)),
                "structured_matches": int(item.get("structured_matches", 0)),
                "support_count": int(item.get("support_count", 0)),
                "source_count": int(item.get("source_count", 0)),
                "token_support": list(item.get("token_support", []))[:6],
                "best_observation": dict(item.get("best_observation") or {}),
            }
            for item in ranked[:10]
        ],
    }


def predict_disc_class_ocr(
    roi_image: np.ndarray,
    fallback_images: Sequence[np.ndarray] | None = None,
    *,
    include_debug: bool = False,
) -> Dict[str, object]:
    if roi_image is None or roi_image.size == 0:
        response = {
            "code": "UNKNOWN",
            "confidence": 0.0,
            "confidence_tier": "failed_unknown",
            "candidates": [],
            "whitelist_candidates": [],
            "raw_ocr_text": "",
            "normalized_text": "",
            "selection_reason": "Disc crop was empty.",
            "decision_source": "ocr_failed",
            "engine": "hybrid-ocr-whitelist",
        }
        if include_debug:
            response["ocr_debug"] = _build_ocr_debug_payload([], [])
        return response

    observations: List[Dict[str, object]] = []
    for observation in _ocr_observations(roi_image):
        observation["source_index"] = 0
        observations.append(observation)
    ranked = _score_candidates(observations)
    if fallback_images and not _is_rank_confident_enough(ranked):
        for source_index, fallback_image in enumerate(fallback_images, start=1):
            if fallback_image is None or fallback_image.size == 0:
                continue
            for observation in _ocr_observations(fallback_image):
                observation["source_index"] = source_index
                observations.append(observation)
            ranked = _score_candidates(observations)
            if _is_rank_confident_enough(ranked):
                break

    if not observations:
        response = {
            "code": "UNKNOWN",
            "confidence": 0.0,
            "confidence_tier": "failed_unknown",
            "candidates": [],
            "whitelist_candidates": [],
            "raw_ocr_text": "",
            "normalized_text": "",
            "selection_reason": "No OCR text could be extracted from the disc crop.",
            "decision_source": "ocr_failed",
            "engine": "hybrid-ocr-whitelist",
        }
        if include_debug:
            response["ocr_debug"] = _build_ocr_debug_payload(observations, ranked)
        return response

    unique_raw = []
    for observation in observations:
        raw_text = str(observation["raw_text"])
        if raw_text not in unique_raw:
            unique_raw.append(raw_text)
        if len(unique_raw) >= 4:
            break

    if not ranked:
        response = {
            "code": "UNKNOWN",
            "confidence": 0.0,
            "confidence_tier": "failed_unknown",
            "candidates": [],
            "whitelist_candidates": [],
            "raw_ocr_text": " | ".join(unique_raw),
            "normalized_text": "",
            "selection_reason": "OCR text was present, but it did not match the approved antibiotic code whitelist safely.",
            "decision_source": "ocr_failed",
            "engine": "hybrid-ocr-whitelist",
        }
        if include_debug:
            response["ocr_debug"] = _build_ocr_debug_payload(observations, ranked)
        return response

    top = ranked[0]
    ranked_lookup = {str(item["code"]): item for item in ranked}
    second_score = float(ranked[1]["score"]) if len(ranked) > 1 else 0.0
    second_exact = int(ranked[1]["exact_matches"]) if len(ranked) > 1 else 0
    margin = float(top["score"]) - second_score
    best_observation = dict(top.get("best_observation") or {})
    strengths = list(best_observation.get("strengths", []))
    strength_fragment = ""
    if strengths:
        strength_fragment = f" using strength {'/'.join(str(value) for value in strengths)}"

    top_code = str(top["code"])
    top_score = float(top["score"])
    top_exact = int(top["exact_matches"]) > 0
    top_alias = int(top.get("alias_matches", 0)) > 0
    top_support_count = int(top["support_count"])
    top_source_count = int(top.get("source_count", 0))
    top_strength_hits = int(top["strength_hits"])
    top_global_strength_matches = int(top.get("global_strength_matches", 0))
    top_structured_matches = int(top.get("structured_matches", 0))
    best_token = str(best_observation.get("token", ""))
    candidates = [str(item["code"]) for item in ranked[:6]]
    single_letter_without_strength = len(top_code) == 1 and not strengths

    confidence_tier = "failed_unknown"
    detected_code = "UNKNOWN"
    selection_reason = "OCR evidence was too weak to return a safe code automatically."
    decision_source = "ocr_failed"

    if top_code == "FOS" and "FOX" in ranked_lookup:
        fox_score = float(ranked_lookup["FOX"]["score"])
        if (top_score - fox_score) <= 0.12 and best_token != "FOS":
            confidence_tier = "uncertain_manual_confirmation_required"
            selection_reason = (
                "Blurred OCR could not safely separate 'FOS' from the nearby whitelist code 'FOX'; manual confirmation is required."
            )
            decision_source = "ocr_suggestion_only"
            response = {
                "code": "UNKNOWN",
                "confidence": round(top_score, 3),
                "confidence_tier": confidence_tier,
                "candidates": candidates,
                "whitelist_candidates": candidates,
                "raw_ocr_text": str(best_observation.get("raw_text", " | ".join(unique_raw))),
                "normalized_text": str(best_observation.get("normalized_text", "")),
                "selection_reason": selection_reason,
                "decision_source": decision_source,
                "engine": "hybrid-ocr-whitelist",
                "observed_strengths": strengths,
            }
            if include_debug:
                response["ocr_debug"] = _build_ocr_debug_payload(observations, ranked)
            return response

    if top_code == "P" and "CIP" in ranked_lookup:
        cip_score = float(ranked_lookup["CIP"]["score"])
        cip_observation = dict(ranked_lookup["CIP"].get("best_observation") or {})
        cip_strengths = list(cip_observation.get("strengths", []))
        if cip_score >= 0.85 and 5 in cip_strengths and not strengths:
            confidence_tier = "probable_match"
            detected_code = "CIP"
            selection_reason = (
                "OCR produced a noisy single-letter 'P', but a stronger whitelist-supported alternative matched 'CIP' with strength 5."
            )
            decision_source = "ocr+whitelist_correction"
            top_code = "CIP"
            top_score = cip_score
            top_exact = int(ranked_lookup["CIP"]["exact_matches"]) > 0
            best_observation = cip_observation
            strengths = cip_strengths
            candidates = [str(item["code"]) for item in ranked[:6]]
            top_alias = int(ranked_lookup["CIP"].get("alias_matches", 0)) > 0
            top_global_strength_matches = int(ranked_lookup["CIP"].get("global_strength_matches", 0))
            top_structured_matches = int(ranked_lookup["CIP"].get("structured_matches", 0))
            top_support_count = int(ranked_lookup["CIP"].get("support_count", top_support_count))
            top_source_count = int(ranked_lookup["CIP"].get("source_count", top_source_count))

    if top_code == "CIP" and "IPM" in ranked_lookup:
        ipm_entry = ranked_lookup["IPM"]
        ipm_score = float(ipm_entry["score"])
        if (
            abs(top_score - ipm_score) <= 0.03
            and int(ipm_entry.get("structured_matches", 0)) > top_structured_matches
            and int(ipm_entry.get("global_strength_matches", 0)) >= 1
        ):
            top_code = "IPM"
            top_score = ipm_score
            top_exact = int(ipm_entry["exact_matches"]) > 0
            top_alias = int(ipm_entry.get("alias_matches", 0)) > 0
            top_support_count = int(ipm_entry.get("support_count", top_support_count))
            top_source_count = int(ipm_entry.get("source_count", top_source_count))
            top_strength_hits = int(ipm_entry.get("strength_hits", top_strength_hits))
            top_global_strength_matches = int(ipm_entry.get("global_strength_matches", top_global_strength_matches))
            top_structured_matches = int(ipm_entry.get("structured_matches", top_structured_matches))
            best_observation = dict(ipm_entry.get("best_observation") or {})
            strengths = list(best_observation.get("strengths", []))
            strength_fragment = ""
            if strengths:
                strength_fragment = f" using strength {'/'.join(str(value) for value in strengths)}"
            candidates = [str(item["code"]) for item in ranked[:6]]

    if len(top_code) == 1 and not strengths and top_support_count < 2:
        confidence_tier = "uncertain_manual_confirmation_required"
        selection_reason = (
            f"OCR only produced the short single-letter token '{top_code}' once; manual confirmation is required."
        )
        decision_source = "ocr_suggestion_only"
        response = {
            "code": "UNKNOWN",
            "confidence": round(top_score, 3),
            "confidence_tier": confidence_tier,
            "candidates": candidates,
            "whitelist_candidates": candidates,
            "raw_ocr_text": str(best_observation.get("raw_text", " | ".join(unique_raw))),
            "normalized_text": str(best_observation.get("normalized_text", "")),
            "selection_reason": selection_reason,
            "decision_source": decision_source,
            "engine": "hybrid-ocr-whitelist",
            "observed_strengths": strengths,
        }
        if include_debug:
            response["ocr_debug"] = _build_ocr_debug_payload(observations, ranked)
        return response

    if (
        top_exact
        and not strengths
        and top_support_count <= 3
        and top_source_count <= 1
        and len(ranked) > 1
        and (
            (
                float(ranked[1]["score"]) >= (top_score - 0.03)
                and int(ranked[1].get("exact_matches", 0)) > 0
            )
            or float(ranked[1]["score"]) >= (top_score - 0.06)
        )
    ):
        confidence_tier = "uncertain_manual_confirmation_required"
        selection_reason = (
            f"OCR produced '{top_code}' from a narrow single-source exact read,"
            " but nearby alternatives remained too close to trust it automatically."
        )
        decision_source = "ocr_suggestion_only"
        response = {
            "code": "UNKNOWN",
            "confidence": round(top_score, 3),
            "confidence_tier": confidence_tier,
            "candidates": candidates,
            "whitelist_candidates": candidates,
            "raw_ocr_text": str(best_observation.get("raw_text", " | ".join(unique_raw))),
            "normalized_text": str(best_observation.get("normalized_text", "")),
            "selection_reason": selection_reason,
            "decision_source": decision_source,
            "engine": "hybrid-ocr-whitelist",
            "observed_strengths": strengths,
        }
        if include_debug:
            response["ocr_debug"] = _build_ocr_debug_payload(observations, ranked)
        return response

    if (
        single_letter_without_strength
        and top_support_count < 4
        and margin < max(settings.OCR_MIN_MARGIN * 0.5, 0.08)
    ):
        confidence_tier = "uncertain_manual_confirmation_required"
        selection_reason = (
            f"OCR repeatedly favored the short single-letter token '{top_code}',"
            " but there was not enough surrounding evidence to trust it automatically."
        )
        decision_source = "ocr_suggestion_only"
    elif (
        top_exact
        and not top_alias
        and not single_letter_without_strength
        and top_score >= settings.OCR_HIGH_CONFIDENCE
        and (
        margin >= settings.OCR_MIN_MARGIN or second_exact == 0
        )
    ):
        confidence_tier = "high_confidence_exact"
        detected_code = top_code
        selection_reason = (
            f"Rotation voting produced an exact whitelist match for '{top_code}'"
            f" from OCR '{best_observation.get('raw_text', '')}'{strength_fragment}."
        )
        decision_source = "ocr+rotation_voting"
    elif top_score >= settings.OCR_MIN_CONFIDENCE and (
        margin >= (settings.OCR_MIN_MARGIN * 0.35) or top_exact or int(top["support_count"]) >= 2
    ):
        if top_alias and top_global_strength_matches == 0:
            confidence_tier = "uncertain_manual_confirmation_required"
            selection_reason = (
                f"OCR only reached '{top_code}' through a confusable alias-like read"
                " without matching strength support; manual confirmation is required."
            )
            decision_source = "ocr_suggestion_only"
        elif single_letter_without_strength and top_support_count < 5:
            confidence_tier = "uncertain_manual_confirmation_required"
            selection_reason = (
                f"OCR favored the single-letter code '{top_code}',"
                " but it still needs confirmation because no strength or longer label structure was recovered."
            )
            decision_source = "ocr_suggestion_only"
        elif len(top_code) <= 2 and not top_exact and top_global_strength_matches == 0:
            confidence_tier = "uncertain_manual_confirmation_required"
            selection_reason = (
                f"OCR favored the short code '{top_code}',"
                " but the evidence stayed partial and did not include a matching strength."
            )
            decision_source = "ocr_suggestion_only"
        elif not top_exact and top_structured_matches == 0 and top_global_strength_matches == 0 and top_support_count < 3:
            confidence_tier = "uncertain_manual_confirmation_required"
            selection_reason = (
                f"OCR evidence for '{top_code}' was too fragmented to trust automatically; manual confirmation is required."
            )
            decision_source = "ocr_suggestion_only"
        elif len(top_code) <= 2 and not top_exact and top_strength_hits == 0 and margin < settings.OCR_MIN_MARGIN:
            confidence_tier = "uncertain_manual_confirmation_required"
            selection_reason = (
                f"OCR favored '{top_code}' only weakly over nearby short-code alternatives; manual confirmation is required."
            )
            decision_source = "ocr_suggestion_only"
        else:
            confidence_tier = "probable_match"
            detected_code = top_code
            selection_reason = (
                f"Rotation voting and whitelist scoring favored '{top_code}' over nearby alternatives"
                f"{strength_fragment}; manual confirmation is still required."
            )
            decision_source = "ocr+whitelist_correction"
    elif top_score >= settings.OCR_SUGGESTION_CONFIDENCE:
        confidence_tier = "uncertain_manual_confirmation_required"
        selection_reason = (
            f"OCR evidence was split; the top suggestion is '{top_code}'{strength_fragment},"
            " but the label must be confirmed manually."
        )
        decision_source = "ocr_suggestion_only"

    response = {
        "code": detected_code,
        "confidence": round(top_score, 3),
        "confidence_tier": confidence_tier,
        "candidates": candidates,
        "whitelist_candidates": candidates,
        "raw_ocr_text": str(best_observation.get("raw_text", " | ".join(unique_raw))),
        "normalized_text": str(best_observation.get("normalized_text", "")),
        "selection_reason": selection_reason,
        "decision_source": decision_source,
        "engine": "hybrid-ocr-whitelist",
        "observed_strengths": strengths,
    }
    if include_debug:
        response["ocr_debug"] = _build_ocr_debug_payload(observations, ranked)
    return response
