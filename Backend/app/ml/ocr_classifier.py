from __future__ import annotations

import re
from itertools import zip_longest
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import pytesseract

from app.core.config import settings

ANTIBIOTIC_CODEBOOK: Dict[str, Dict[str, object]] = {
    "P": {"strengths": {10}},
    "AMP": {"strengths": {10}},
    "AM": {"strengths": {10}},
    "AMC": {"strengths": {30}},
    "SAM": {"strengths": {20}},
    "TZP": {"strengths": {110}},
    "PIP": {"strengths": {100}},
    "CAZ": {"strengths": {30}},
    "CTX": {"strengths": {30}},
    "CRO": {"strengths": {30}},
    "CEF": {"strengths": {30}},
    "CFM": {"strengths": {5}},
    "CPD": {"strengths": {10}},
    "CFX": {"strengths": {30}},
    "FOX": {"strengths": {30}},
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
    "KAN": {"strengths": {30}},
    "STR": {"strengths": {10}},
    "NET": {"strengths": {30}},
    "CIP": {"strengths": {5}},
    "NOR": {"strengths": {10}},
    "LEV": {"strengths": {5}},
    "OFX": {"strengths": {5}},
    "NAL": {"strengths": {30}},
    "GAT": {"strengths": {5}},
    "MOX": {"strengths": {5}},
    "TE": {"strengths": {30}},
    "DO": {"strengths": {30}},
    "TGC": {"strengths": {15}},
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
    "MUP": {"strengths": {5, 200}},
    "COL": {"strengths": {10}},
    "POL": {"strengths": {300}},
    "NIT": {"strengths": {300}},
    "FUR": {"strengths": {100}},
}

DIRECT_ALIAS_MAP = {
    "GN": "CN",
    "GEN": "CN",
    "IE": "TE",
    "SAN": "SAM",
    "AN": "AM",
    "AK": "AMK",
    "CIF": "CIP",
}

VALID_STRENGTHS = {2, 5, 10, 15, 20, 25, 30, 50, 100, 110, 200, 300}
PSM_CONFIGS: Sequence[Tuple[int, str]] = (
    (6, "--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    (11, "--psm 11 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
)
VARIANT_WEIGHTS = {
    "sharpened": 1.0,
    "blackhat": 1.08,
    "inv_otsu": 1.04,
    "dilate": 1.02,
    "adaptive": 0.98,
}
PSM_WEIGHTS = {6: 1.0, 11: 0.97}
ANGLE_OFFSETS = (-12, 0, 12)
FALLBACK_ANGLES = (0, 45, 90, 135, 180, 225, 270, 315)
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


def _estimate_candidate_angles(blackhat: np.ndarray) -> List[int]:
    percentile = int(np.percentile(blackhat, 97))
    mask = cv2.inRange(blackhat, percentile, 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    coords = cv2.findNonZero(mask)
    if coords is None or len(coords) < 8:
        return list(FALLBACK_ANGLES)

    rect = cv2.minAreaRect(coords)
    base_angle = float(rect[-1]) % 360.0
    angle_set = set()
    for base in (base_angle, base_angle + 90.0, base_angle + 180.0, base_angle + 270.0):
        for offset in ANGLE_OFFSETS:
            angle_set.add(int(round((base + offset) % 360.0)))
    return sorted(angle_set)


def _extract_code_tokens(raw_text: str) -> List[str]:
    tokens = re.findall(r"[A-Z0-9]{1,6}", raw_text.upper())
    code_tokens: List[str] = []
    seen = set()
    for token in tokens:
        if token.isdigit():
            continue
        normalized = token.translate(OCR_CHAR_TRANSLATION)
        normalized = re.sub(r"[^A-Z]", "", normalized)
        if not normalized:
            continue
        normalized = normalized[:4]
        if normalized not in seen:
            seen.add(normalized)
            code_tokens.append(normalized)
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


def _alignment_score(token: str, candidate: str) -> Tuple[float, str]:
    alias_target = DIRECT_ALIAS_MAP.get(token)
    if token == candidate:
        return 1.0, "exact"
    if alias_target == candidate:
        return 0.95, "alias"
    if len(token) == 1 and len(candidate) > 1:
        return 0.0, "too_short"
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


def _ocr_observations(roi_image: np.ndarray) -> List[Dict[str, object]]:
    variants = _build_variants(roi_image)
    candidate_angles = _estimate_candidate_angles(variants["blackhat"])
    observations: List[Dict[str, object]] = []

    for variant_name, variant_image in variants.items():
        for angle in candidate_angles:
            rotated = _rotate_image(variant_image, angle)
            for psm, config in PSM_CONFIGS:
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
    return observations


def _score_candidates(observations: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    aggregate: Dict[str, Dict[str, object]] = {
        code: {
            "code": code,
            "scores": [],
            "exact_matches": 0,
            "strength_hits": 0,
            "angles": set(),
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
            entry["reasons"].append(reason)
            if strength_delta > 0:
                entry["strength_hits"] += 1
            if token == candidate or DIRECT_ALIAS_MAP.get(token) == candidate:
                entry["exact_matches"] += 1
            if total > float(entry["best_score"]):
                entry["best_score"] = total
                entry["best_observation"] = {
                    "raw_text": observation["raw_text"],
                    "normalized_text": observation["normalized_text"],
                    "angle": observation["angle"],
                    "variant": observation["variant"],
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
        combined_score = min(0.99, (best_score * 0.72) + (mean_top * 0.28) + support_bonus + angle_bonus)
        if int(entry["exact_matches"]) == 0:
            combined_score = min(combined_score, 0.94)
        if len(scores) == 1 and int(entry["exact_matches"]) == 0:
            combined_score = max(0.0, combined_score - 0.06)
        if len(str(entry["code"])) == 1 and int(entry["exact_matches"]) < 2:
            combined_score = min(combined_score, 0.84)
        ranked.append(
            {
                "code": entry["code"],
                "score": round(combined_score, 3),
                "exact_matches": int(entry["exact_matches"]),
                "strength_hits": int(entry["strength_hits"]),
                "support_count": len(scores),
                "best_observation": entry["best_observation"],
            }
        )

    ranked.sort(
        key=lambda item: (
            -float(item["score"]),
            -int(item["strength_hits"]),
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


def predict_disc_class_ocr(roi_image: np.ndarray) -> Dict[str, object]:
    if roi_image is None or roi_image.size == 0:
        return {
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

    observations = _ocr_observations(roi_image)
    if not observations:
        return {
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

    ranked = _score_candidates(observations)
    unique_raw = []
    for observation in observations:
        raw_text = str(observation["raw_text"])
        if raw_text not in unique_raw:
            unique_raw.append(raw_text)
        if len(unique_raw) >= 4:
            break

    if not ranked:
        return {
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
    best_token = str(best_observation.get("token", ""))
    candidates = [str(item["code"]) for item in ranked[:6]]

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
            return {
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

    if top_exact and top_score >= settings.OCR_HIGH_CONFIDENCE and (
        margin >= settings.OCR_MIN_MARGIN or second_exact == 0
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

    return {
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
