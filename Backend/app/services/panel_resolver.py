from __future__ import annotations

from difflib import SequenceMatcher
from itertools import permutations
from math import hypot
from typing import Iterable, Sequence

from app.models.schemas import DiscMeasurement

STANDARD_SIX_DISC_PANEL = (
    {"code": "SXT", "position": (0.44, 0.21), "rank": "upper"},
    {"code": "FOX", "position": (0.72, 0.27), "rank": "upper"},
    {"code": "P", "position": (0.24, 0.42), "rank": "smallest"},
    {"code": "TE", "position": (0.55, 0.47), "rank": "middle"},
    {"code": "DO", "position": (0.84, 0.54), "rank": "middle"},
    {"code": "LZD", "position": (0.32, 0.71), "rank": "largest"},
)


def _label_similarity(result: DiscMeasurement, panel_code: str) -> float:
    tokens = [result.detected_code] + list(result.label_candidates)
    best = 0.0
    for token in tokens:
        if not token:
            continue
        token = token.upper()
        score = SequenceMatcher(a=token, b=panel_code).ratio()
        if token == panel_code:
            score += 0.2
        elif panel_code in token or token in panel_code:
            score += 0.08
        best = max(best, min(1.0, score))
    return best


def _measurement_rank_cost(index_desc: int, total: int, rule: str) -> float:
    if rule == "largest":
        return float(index_desc) * 6.0
    if rule == "smallest":
        return float(max(0, total - 1 - index_desc)) * 6.0
    if rule == "upper":
        return float(abs(index_desc - 1)) * 2.0
    if rule == "middle":
        center_rank = max(1, total // 2)
        return float(abs(index_desc - center_rank)) * 1.6
    return 0.0


def _assignment_cost(result: DiscMeasurement, panel_item: dict, image_width: float, image_height: float, mm_desc_order: Sequence[str]) -> float:
    nx = result.center.x / max(1.0, image_width)
    ny = result.center.y / max(1.0, image_height)
    px, py = panel_item["position"]
    position_distance = hypot(nx - px, ny - py)
    similarity = _label_similarity(result, panel_item["code"])
    rank_index = mm_desc_order.index(result.disc_id)
    measurement_cost = _measurement_rank_cost(rank_index, len(mm_desc_order), panel_item["rank"])
    return (position_distance * 120.0) + ((1.0 - similarity) * 18.0) + measurement_cost


def apply_standard_panel_resolution(results: Sequence[DiscMeasurement], image_shape: tuple[int, int, int] | tuple[int, int]) -> None:
    if len(results) != len(STANDARD_SIX_DISC_PANEL):
        return

    image_height = float(image_shape[0])
    image_width = float(image_shape[1])
    measurement_order = [item.disc_id for item in sorted(results, key=lambda item: item.auto_diameter_mm, reverse=True)]

    best_perm = None
    best_cost = float("inf")
    for assignment in permutations(STANDARD_SIX_DISC_PANEL, len(results)):
        total_cost = 0.0
        for result, panel_item in zip(results, assignment):
            total_cost += _assignment_cost(result, panel_item, image_width, image_height, measurement_order)
        if total_cost < best_cost:
            best_cost = total_cost
            best_perm = assignment

    if best_perm is None:
        return

    average_cost = best_cost / float(len(results))
    if average_cost > 24.0:
        return

    for result, panel_item in zip(results, best_perm):
        similarity = _label_similarity(result, panel_item["code"])
        nx = result.center.x / max(1.0, image_width)
        ny = result.center.y / max(1.0, image_height)
        position_distance = hypot(nx - panel_item["position"][0], ny - panel_item["position"][1])
        resolved_confidence = max(0.48, min(0.82, 0.82 - position_distance * 1.6 + similarity * 0.08))

        should_override = result.detected_code not in {panel_item["code"]} and (
            result.label_confidence < 0.9 or panel_item["code"] in result.label_candidates or similarity >= 0.35
        )
        if not should_override:
            continue

        result.detected_code = panel_item["code"]
        result.final_code = panel_item["code"]
        result.label_engine = "ocr+panel-layout"
        result.label_confidence = round(resolved_confidence, 3)
        result.overall_confidence = round((result.measurement_confidence * 0.55) + (resolved_confidence * 0.45), 3)
        candidates = [panel_item["code"]] + [candidate for candidate in result.label_candidates if candidate != panel_item["code"]]
        result.label_candidates = candidates[:5]
        if "Panel-layout resolver suggested the code; confirm if the label is unclear." not in result.warnings:
            result.warnings.append("Panel-layout resolver suggested the code; confirm if the label is unclear.")
        result.review_required = True
        if result.status == "auto":
            result.status = "review_required"
