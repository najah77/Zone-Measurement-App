from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from math import hypot
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.models.schemas import DiscMeasurement

STANDARD_SIX_DISC_PANEL = (
    {"code": "SXT", "position": (0.44, 0.21), "rank": "upper"},
    {"code": "FOX", "position": (0.72, 0.27), "rank": "upper"},
    {"code": "P", "position": (0.24, 0.42), "rank": "smallest"},
    {"code": "TE", "position": (0.55, 0.47), "rank": "middle"},
    {"code": "DO", "position": (0.84, 0.54), "rank": "middle"},
    {"code": "LZD", "position": (0.32, 0.71), "rank": "largest"},
)

STANDARD_FIVE_DISC_PANEL = (
    {"code": "FOX", "position": (0.36, 0.30), "rank": "middle"},
    {"code": "LZD", "position": (0.65, 0.29), "rank": "largest"},
    {"code": "TE", "position": (0.28, 0.57), "rank": "smallest"},
    {"code": "CN", "position": (0.78, 0.57), "rank": "middle"},
    {"code": "CIP", "position": (0.56, 0.78), "rank": "middle"},
)

STANDARD_SIXTEEN_DISC_PANEL_FF = (
    {"code": "AM10", "position": (0.11, 0.12), "rank": "smallest"},
    {"code": "FEP", "position": (0.37, 0.12), "rank": "largest"},
    {"code": "FOX", "position": (0.63, 0.12), "rank": "middle"},
    {"code": "MEM", "position": (0.89, 0.12), "rank": "largest"},
    {"code": "CPD", "position": (0.11, 0.38), "rank": "middle"},
    {"code": "AMC", "position": (0.37, 0.38), "rank": "middle"},
    {"code": "CRO", "position": (0.63, 0.38), "rank": "largest"},
    {"code": "TPZ", "position": (0.89, 0.38), "rank": "middle"},
    {"code": "CAZ", "position": (0.11, 0.64), "rank": "middle"},
    {"code": "FF", "position": (0.37, 0.64), "rank": "largest"},
    {"code": "F100", "position": (0.63, 0.64), "rank": "middle"},
    {"code": "CN", "position": (0.89, 0.64), "rank": "middle"},
    {"code": "ETP", "position": (0.11, 0.89), "rank": "largest"},
    {"code": "CIP", "position": (0.37, 0.89), "rank": "largest"},
    {"code": "PEF", "position": (0.63, 0.89), "rank": "middle"},
    {"code": "SXT", "position": (0.89, 0.89), "rank": "middle"},
)

STANDARD_SIXTEEN_DISC_PANEL_TOB = (
    {"code": "AM10", "position": (0.11, 0.12), "rank": "smallest"},
    {"code": "FEP", "position": (0.37, 0.12), "rank": "largest"},
    {"code": "FOX", "position": (0.63, 0.12), "rank": "smallest"},
    {"code": "MEM", "position": (0.89, 0.12), "rank": "largest"},
    {"code": "CPD", "position": (0.11, 0.38), "rank": "middle"},
    {"code": "AMC", "position": (0.37, 0.38), "rank": "smallest"},
    {"code": "CRO", "position": (0.63, 0.38), "rank": "largest"},
    {"code": "TPZ", "position": (0.89, 0.38), "rank": "middle"},
    {"code": "CAZ", "position": (0.11, 0.64), "rank": "middle"},
    {"code": "AK", "position": (0.37, 0.64), "rank": "middle"},
    {"code": "TOB", "position": (0.63, 0.64), "rank": "middle"},
    {"code": "CN", "position": (0.89, 0.64), "rank": "middle"},
    {"code": "ETP", "position": (0.11, 0.89), "rank": "largest"},
    {"code": "CIP", "position": (0.37, 0.89), "rank": "largest"},
    {"code": "PEF", "position": (0.63, 0.89), "rank": "smallest"},
    {"code": "SXT", "position": (0.89, 0.89), "rank": "smallest"},
)

STANDARD_SIXTEEN_DISC_PANEL_MIN = (
    {"code": "AMC", "position": (0.11, 0.12), "rank": "smallest"},
    {"code": "FEP", "position": (0.37, 0.12), "rank": "smallest"},
    {"code": "TGC", "position": (0.63, 0.12), "rank": "smallest"},
    {"code": "IPM", "position": (0.89, 0.12), "rank": "middle"},
    {"code": "CAZ", "position": (0.11, 0.38), "rank": "smallest"},
    {"code": "TPZ", "position": (0.37, 0.38), "rank": "middle"},
    {"code": "SAM", "position": (0.63, 0.38), "rank": "smallest"},
    {"code": "MEM", "position": (0.89, 0.38), "rank": "middle"},
    {"code": "ATM", "position": (0.11, 0.64), "rank": "middle"},
    {"code": "CIP", "position": (0.37, 0.64), "rank": "largest"},
    {"code": "LEV", "position": (0.63, 0.64), "rank": "middle"},
    {"code": "MIN", "position": (0.89, 0.64), "rank": "middle"},
    {"code": "TOB", "position": (0.11, 0.89), "rank": "middle"},
    {"code": "AK", "position": (0.37, 0.89), "rank": "middle"},
    {"code": "CN", "position": (0.63, 0.89), "rank": "middle"},
    {"code": "SXT", "position": (0.89, 0.89), "rank": "smallest"},
)

STANDARD_SIXTEEN_DISC_PANELS = (
    STANDARD_SIXTEEN_DISC_PANEL_FF,
    STANDARD_SIXTEEN_DISC_PANEL_TOB,
    STANDARD_SIXTEEN_DISC_PANEL_MIN,
)

CANONICAL_CODE_ALIASES = {
    "AM": "AM10",
    "TOP": "TOB",
}


def _canonical_code(code: str | None) -> str:
    token = (code or "").upper().strip()
    return CANONICAL_CODE_ALIASES.get(token, token)


def _codes_equivalent(left: str | None, right: str | None) -> bool:
    return bool(_canonical_code(left)) and _canonical_code(left) == _canonical_code(right)


def _label_similarity(result: DiscMeasurement, panel_code: str) -> float:
    tokens = [result.detected_code] + list(result.label_candidates)
    best = 0.0
    for token in tokens:
        if not token:
            continue
        token = token.upper()
        score = SequenceMatcher(a=token, b=panel_code).ratio()
        if _codes_equivalent(token, panel_code):
            score = max(score, 0.96)
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


def _assignment_cost(
    result: DiscMeasurement,
    panel_item: dict,
    image_width: float,
    image_height: float,
    mm_desc_order: Sequence[str],
    *,
    measurement_weight: float = 1.0,
) -> float:
    nx = result.center.x / max(1.0, image_width)
    ny = result.center.y / max(1.0, image_height)
    px, py = panel_item["position"]
    position_distance = hypot(nx - px, ny - py)
    similarity = _label_similarity(result, panel_item["code"])
    rank_index = mm_desc_order.index(result.disc_id)
    measurement_cost = _measurement_rank_cost(rank_index, len(mm_desc_order), panel_item["rank"]) * measurement_weight
    return (position_distance * 120.0) + ((1.0 - similarity) * 18.0) + measurement_cost


def _resolve_panel_assignment(
    results: Sequence[DiscMeasurement],
    image_shape: tuple[int, int, int] | tuple[int, int],
    panel_definition: Sequence[dict],
    *,
    max_average_cost: float,
) -> tuple[Sequence[dict] | None, float]:
    if len(results) != len(panel_definition):
        return None, float("inf")

    image_height = float(image_shape[0])
    image_width = float(image_shape[1])
    measurement_order = [item.disc_id for item in sorted(results, key=lambda item: item.auto_diameter_mm, reverse=True)]

    cost_matrix = np.zeros((len(results), len(panel_definition)), dtype=np.float32)
    for row_index, result in enumerate(results):
        for col_index, panel_item in enumerate(panel_definition):
            cost_matrix[row_index, col_index] = _assignment_cost(
                result,
                panel_item,
                image_width,
                image_height,
                measurement_order,
            )

    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    if len(row_indices) != len(results):
        return None, float("inf")

    ordered_assignment: list[dict] = [panel_definition[0]] * len(results)
    total_cost = 0.0
    for row_index, col_index in zip(row_indices, col_indices):
        ordered_assignment[int(row_index)] = panel_definition[int(col_index)]
        total_cost += float(cost_matrix[int(row_index), int(col_index)])

    average_cost = total_cost / float(len(results))
    if average_cost > max_average_cost:
        return None, average_cost
    return tuple(ordered_assignment), average_cost


def _apply_panel_resolution(
    results: Sequence[DiscMeasurement],
    image_shape: tuple[int, int, int] | tuple[int, int],
    panel_definition: Sequence[dict],
    *,
    max_average_cost: float,
) -> None:
    image_height = float(image_shape[0])
    image_width = float(image_shape[1])
    best_perm, average_cost = _resolve_panel_assignment(
        results,
        image_shape,
        panel_definition,
        max_average_cost=max_average_cost,
    )
    if best_perm is None:
        return
    _apply_panel_items(results, image_shape, best_perm)


def _apply_panel_items(
    results: Sequence[DiscMeasurement],
    image_shape: tuple[int, int, int] | tuple[int, int],
    panel_items: Sequence[dict],
) -> None:
    image_height = float(image_shape[0])
    image_width = float(image_shape[1])
    panel_code_counts = Counter(_canonical_code(item["code"]) for item in panel_items)
    detected_code_counts = Counter(
        _canonical_code(item.detected_code)
        for item in results
        if _canonical_code(item.detected_code) and _canonical_code(item.detected_code) != "UNKNOWN"
    )

    for result, panel_item in zip(results, panel_items):
        similarity = _label_similarity(result, panel_item["code"])
        nx = result.center.x / max(1.0, image_width)
        ny = result.center.y / max(1.0, image_height)
        position_distance = hypot(nx - panel_item["position"][0], ny - panel_item["position"][1])
        resolved_confidence = max(0.48, min(0.82, 0.82 - position_distance * 1.6 + similarity * 0.08))
        panel_code = panel_item["code"]
        panel_canonical = _canonical_code(panel_code)
        detected_canonical = _canonical_code(result.detected_code)
        candidate_canonicals = {_canonical_code(candidate) for candidate in result.label_candidates}
        panel_missing = detected_code_counts.get(panel_canonical, 0) < panel_code_counts.get(panel_canonical, 0)
        detected_outside_panel = bool(detected_canonical) and detected_canonical not in panel_code_counts
        detected_duplicate = bool(detected_canonical) and detected_code_counts.get(detected_canonical, 0) > panel_code_counts.get(detected_canonical, 0)
        panel_supported_by_candidates = panel_canonical in candidate_canonicals
        dense_panel = len(results) >= 8
        result.layout_suggestion = panel_code

        if panel_code not in result.label_candidates:
            result.label_candidates = [panel_code, *result.label_candidates][:8]
        if panel_code not in result.whitelist_candidates_considered:
            result.whitelist_candidates_considered = [panel_code, *result.whitelist_candidates_considered][:8]

        if _codes_equivalent(result.detected_code, panel_code):
            result.final_code = panel_code
            continue

        if result.detected_code == "UNKNOWN" and result.label_confidence_tier in {
            "failed_unknown",
            "uncertain_manual_confirmation_required",
        }:
            result.detected_code = panel_code
            result.final_code = panel_code
            result.label_engine = f"{result.label_engine}+panel-layout"
            result.label_decision_source = "panel_layout_assist"
            result.label_confidence = round(max(result.label_confidence, resolved_confidence), 3)
            result.label_confidence_tier = "uncertain_manual_confirmation_required"
            result.label_selection_reason = (
                f"Panel layout suggested '{panel_code}' because OCR was unclear; manual confirmation is still required."
            )
            result.overall_confidence = round((result.measurement_confidence * 0.55) + (result.label_confidence * 0.45), 3)
            if "Panel layout suggested the code because OCR was unclear; confirm it manually." not in result.warnings:
                result.warnings.append("Panel layout suggested the code because OCR was unclear; confirm it manually.")
            result.review_required = True
            if result.status == "auto":
                result.status = "review_required"
            continue

        if (
            dense_panel
            and position_distance <= 0.085
            and panel_missing
            and (detected_outside_panel or detected_duplicate or panel_supported_by_candidates)
        ):
            original_code = result.detected_code
            result.detected_code = panel_code
            result.final_code = panel_code
            result.label_engine = f"{result.label_engine}+panel-layout"
            result.label_decision_source = "panel_layout_consistency"
            result.label_confidence = round(max(result.label_confidence, resolved_confidence), 3)
            result.label_confidence_tier = (
                "uncertain_manual_confirmation_required"
                if result.label_confidence_tier == "high_confidence_exact" and not panel_supported_by_candidates
                else "probable_match"
            )
            result.label_selection_reason = (
                f"Dense standard-panel consistency promoted '{panel_code}'"
                f" because '{original_code}' would leave the panel with duplicate or missing codes."
            )
            result.overall_confidence = round((result.measurement_confidence * 0.55) + (result.label_confidence * 0.45), 3)
            warning = (
                f"Dense panel consistency promoted '{panel_code}' instead of '{original_code}'; confirm the label manually."
            )
            if warning not in result.warnings:
                result.warnings.append(warning)
            result.review_required = True
            if result.status in {"auto", "failed"}:
                result.status = "review_required"
            continue

        if (
            len(result.detected_code) <= 1
            and (
                panel_code in result.label_candidates
                or (
                    dense_panel
                    and position_distance <= 0.08
                    and result.label_confidence_tier in {"high_confidence_exact", "probable_match", "uncertain_manual_confirmation_required"}
                )
            )
        ):
            original_code = result.detected_code
            result.detected_code = panel_code
            result.final_code = panel_code
            result.label_engine = f"{result.label_engine}+panel-layout"
            result.label_decision_source = "ocr+panel_layout_disambiguation"
            result.label_confidence = round(max(result.label_confidence, resolved_confidence + (0.05 if dense_panel else 0.0)), 3)
            result.label_confidence_tier = "probable_match"
            result.label_selection_reason = (
                f"Panel layout promoted '{panel_code}' because OCR only returned the short code '{original_code}'."
            )
            result.overall_confidence = round((result.measurement_confidence * 0.55) + (result.label_confidence * 0.45), 3)
            warning = (
                f"Panel layout promoted '{panel_code}' because the OCR read was only '{original_code}'; confirm it manually."
            )
            if warning not in result.warnings:
                result.warnings.append(warning)
            result.review_required = True
            if result.status in {"auto", "failed"}:
                result.status = "review_required"
            continue

        if (
            result.label_confidence_tier != "high_confidence_exact"
            and panel_code in result.label_candidates
            and result.detected_code != panel_code
        ):
            original_code = result.detected_code
            result.detected_code = panel_code
            result.final_code = panel_code
            result.label_engine = f"{result.label_engine}+panel-layout"
            result.label_decision_source = "ocr+panel_layout_disambiguation"
            result.label_confidence = round(max(result.label_confidence, resolved_confidence), 3)
            result.label_confidence_tier = "uncertain_manual_confirmation_required"
            result.label_selection_reason = (
                f"Panel layout promoted '{panel_code}' because OCR only produced a non-exact probable match for '{original_code}'."
            )
            result.overall_confidence = round((result.measurement_confidence * 0.55) + (result.label_confidence * 0.45), 3)
            warning = (
                f"Panel layout promoted '{panel_code}' because OCR favored '{original_code}' only weakly; confirm it manually."
            )
            if warning not in result.warnings:
                result.warnings.append(warning)
            result.review_required = True
            if result.status in {"auto", "failed"}:
                result.status = "review_required"
            continue

        if (
            dense_panel
            and position_distance <= 0.065
            and result.detected_code != panel_code
            and result.label_confidence_tier in {"failed_unknown", "uncertain_manual_confirmation_required"}
        ):
            original_code = result.detected_code
            result.detected_code = panel_code
            result.final_code = panel_code
            result.label_engine = f"{result.label_engine}+panel-layout"
            result.label_decision_source = "panel_layout_assist"
            result.label_confidence = round(max(result.label_confidence, resolved_confidence), 3)
            result.label_confidence_tier = "uncertain_manual_confirmation_required"
            result.label_selection_reason = (
                f"Panel layout suggested '{panel_code}' because OCR was too ambiguous for a dense standard panel."
            )
            result.overall_confidence = round((result.measurement_confidence * 0.55) + (result.label_confidence * 0.45), 3)
            warning = (
                f"Panel layout suggested '{panel_code}' because OCR returned '{original_code}' too ambiguously; confirm it manually."
            )
            if warning not in result.warnings:
                result.warnings.append(warning)
            result.review_required = True
            if result.status in {"auto", "failed"}:
                result.status = "review_required"
            continue

        disagreement_warning = (
            f"Panel layout suggests '{panel_code}', but OCR read '{result.detected_code}'; confirm the disc code manually."
        )
        if disagreement_warning not in result.warnings:
            result.warnings.append(disagreement_warning)
        result.review_required = True
        if result.status in {"auto", "failed"}:
            result.status = "review_required"


def _dense_grid_order(results: Sequence[DiscMeasurement], *, columns: int = 4) -> list[DiscMeasurement]:
    ordered_by_y = sorted(results, key=lambda item: (item.center.y, item.center.x))
    grid_order: list[DiscMeasurement] = []
    for index in range(0, len(ordered_by_y), columns):
        row = ordered_by_y[index : index + columns]
        grid_order.extend(sorted(row, key=lambda item: item.center.x))
    return grid_order


def _dense_panel_average_cost(
    results: Sequence[DiscMeasurement],
    image_shape: tuple[int, int, int] | tuple[int, int],
    panel_definition: Sequence[dict],
) -> float:
    if len(results) != len(panel_definition):
        return float("inf")
    ordered_results = _dense_grid_order(results)
    image_height = float(image_shape[0])
    image_width = float(image_shape[1])
    measurement_order = [item.disc_id for item in sorted(results, key=lambda item: item.auto_diameter_mm, reverse=True)]
    total_cost = 0.0
    for result, panel_item in zip(ordered_results, panel_definition):
        total_cost += _assignment_cost(
            result,
            panel_item,
            image_width,
            image_height,
            measurement_order,
            measurement_weight=0.0,
        )
    return total_cost / float(len(ordered_results))


def _dense_panel_support_metrics(
    results: Sequence[DiscMeasurement],
    panel_definition: Sequence[dict],
) -> tuple[int, int, int]:
    if len(results) != len(panel_definition):
        return (0, 0, 0)

    ordered_results = _dense_grid_order(results)
    exact_matches = 0
    candidate_hits = 0
    high_confidence_conflicts = 0

    for result, panel_item in zip(ordered_results, panel_definition):
        panel_code = panel_item["code"]
        candidate_canonicals = {_canonical_code(candidate) for candidate in result.label_candidates}
        if _codes_equivalent(result.detected_code, panel_code):
            exact_matches += 1
        elif _canonical_code(panel_code) in candidate_canonicals:
            candidate_hits += 1

        if result.label_confidence_tier == "high_confidence_exact" and not _codes_equivalent(result.detected_code, panel_code):
            high_confidence_conflicts += 1

    return exact_matches, candidate_hits, high_confidence_conflicts


def _apply_dense_panel_resolution(
    results: Sequence[DiscMeasurement],
    image_shape: tuple[int, int, int] | tuple[int, int],
    panel_definition: Sequence[dict],
    *,
    max_average_cost: float,
) -> None:
    ordered_results = _dense_grid_order(results)
    average_cost = _dense_panel_average_cost(ordered_results, image_shape, panel_definition)
    if average_cost > max_average_cost:
        return
    _apply_panel_items(ordered_results, image_shape, panel_definition)


def apply_standard_panel_resolution(results: Sequence[DiscMeasurement], image_shape: tuple[int, int, int] | tuple[int, int]) -> None:
    if len(results) == len(STANDARD_SIX_DISC_PANEL):
        _apply_panel_resolution(results, image_shape, STANDARD_SIX_DISC_PANEL, max_average_cost=24.0)
    elif len(results) == len(STANDARD_FIVE_DISC_PANEL):
        _apply_panel_resolution(results, image_shape, STANDARD_FIVE_DISC_PANEL, max_average_cost=18.0)
    elif len(results) == 16:
        best_panel = None
        best_rank: tuple[float, int, int, int] | None = None
        for panel_definition in STANDARD_SIXTEEN_DISC_PANELS:
            average_cost = _dense_panel_average_cost(results, image_shape, panel_definition)
            exact_matches, candidate_hits, conflicts = _dense_panel_support_metrics(results, panel_definition)
            ranking = (
                round(average_cost - (exact_matches * 2.8) - (candidate_hits * 0.9) + (conflicts * 3.2), 6),
                -exact_matches,
                conflicts,
                -candidate_hits,
            )
            if best_rank is None or ranking < best_rank:
                best_panel = panel_definition
                best_rank = ranking
        if best_panel is not None:
            _apply_dense_panel_resolution(results, image_shape, best_panel, max_average_cost=44.5)
