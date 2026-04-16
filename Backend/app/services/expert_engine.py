from __future__ import annotations

from typing import Dict, List

from app.models.schemas import DiscMeasurement


def validate_results(results: List[DiscMeasurement]) -> Dict[str, str]:
    if not results:
        return {"status": "FAILED", "reason": "No discs were detected."}

    review_required = [result for result in results if result.review_required]
    failed = [result for result in results if result.status == "failed"]

    if failed:
        return {
            "status": "REQUIRES_MANUAL_REVIEW",
            "reason": f"{len(failed)} disc(s) could not be measured confidently.",
        }

    if review_required:
        return {
            "status": "REQUIRES_MANUAL_REVIEW",
            "reason": f"{len(review_required)} disc(s) need operator confirmation.",
        }

    return {"status": "VALID", "reason": "All discs passed automatic validation."}
