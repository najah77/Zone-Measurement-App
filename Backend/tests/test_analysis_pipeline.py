from __future__ import annotations

import asyncio
import json
import sys
import unittest
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.append(str(TESTS_DIR))

from app.main import app
from app.ml.ocr_classifier import predict_disc_class_ocr
from app.models.schemas import DiscMeasurement, Point
from app.services.ast_processor import process_ast_image
from app.services.expert_engine import validate_results
from app.utils.calibration import build_calibration, pixels_to_mm
from app.utils.disc_detector import detect_discs
from app.utils.disc_detector import validate_disc_geometry
from app.utils.disc_roi import extract_disc_roi
from app.utils.measurement import apply_no_zone_rule, calculate_inhibition_result
from app.utils.zone_detector import measure_zone
from synthetic_plate import DiscSpec, encode_png, make_label_crop, make_plate_image

REAL_SAMPLE_PATH = ROOT / "data" / "analysis_runs" / "03315fd2-d572-4120-9813-57e6919fcada" / "original_upload.bin"


def _resolve_real_sample_path(preferred_filenames: set[str], fallback: Path) -> Path:
    analysis_root = ROOT / "data" / "analysis_runs"
    candidates: list[tuple[float, Path]] = []
    for analysis_json in analysis_root.glob("*/analysis.json"):
        try:
            payload = json.loads(analysis_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        image_filename = (
            payload.get("analysis", {}).get("image_filename")
            or payload.get("storage", {}).get("image_filename")
            or ""
        )
        if image_filename not in preferred_filenames:
            continue
        upload_path = analysis_json.parent / "original_upload.bin"
        if not upload_path.exists():
            continue
        candidates.append((analysis_json.stat().st_mtime, upload_path))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]
    return fallback


LATEST_SAMPLE_PATH = _resolve_real_sample_path(
    {"test1.jpeg", "latest.jpeg"},
    ROOT / "data" / "analysis_runs" / "4dba7418-2c7a-47b1-9806-bc93a78be306" / "original_upload.bin",
)
REAL_SAMPLE_PATH = _resolve_real_sample_path(
    {"sample.jpeg", "sample_plate.jpeg"},
    REAL_SAMPLE_PATH,
)


@lru_cache(maxsize=4)
def _analyze_fixture(path_str: str, filename: str):
    path = Path(path_str)
    return asyncio.run(
        process_ast_image(
            path.read_bytes(),
            image_filename=filename,
            include_debug_artifacts=False,
        )
    )


def _load_latest_sample_fox_crop() -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(LATEST_SAMPLE_PATH.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Latest sample fixture could not be decoded.")

    from app.utils.plate_extractor import extract_plate

    plate_image, _ = extract_plate(image)
    discs = sorted(detect_discs(plate_image), key=lambda disc: (disc[1], disc[0]))
    if not discs:
        raise RuntimeError("No discs were found in the latest sample fixture.")
    top_left = min(discs, key=lambda disc: disc[0] + disc[1])
    return extract_disc_roi(
        plate_image,
        float(top_left[0]),
        float(top_left[1]),
        float(top_left[2]),
        expand_ratio=0.24,
        mask_scale=0.8,
    )


class GeometryAndNormalizationTests(unittest.TestCase):
    def test_pixels_to_mm_uses_disc_reference(self) -> None:
        calibration = build_calibration([(0.0, 0.0, 20.0), (10.0, 10.0, 21.0), (20.0, 20.0, 19.0)])
        self.assertAlmostEqual(calibration.mm_per_pixel, 0.15, places=2)
        self.assertAlmostEqual(pixels_to_mm(40, calibration.mm_per_pixel), 6.0, places=1)

    def test_no_zone_rule_returns_six_mm(self) -> None:
        self.assertEqual(apply_no_zone_rule(None), 6.0)
        self.assertEqual(apply_no_zone_rule(0), 6.0)
        self.assertEqual(calculate_inhibition_result(0.0, 40.0, 0.15), 6.0)

    def test_disc_geometry_validation_rejects_inconsistent_radii(self) -> None:
        self.assertTrue(validate_disc_geometry([(0.0, 0.0, 20.0), (1.0, 1.0, 21.0), (2.0, 2.0, 19.5)]))
        self.assertFalse(validate_disc_geometry([(0.0, 0.0, 12.0), (1.0, 1.0, 24.0), (2.0, 2.0, 36.0)]))

    def test_confidence_validation_marks_review_required(self) -> None:
        label_crop = make_label_crop("LZD")
        prediction = predict_disc_class_ocr(label_crop)
        self.assertEqual(prediction["code"], "LZD")
        self.assertGreaterEqual(prediction["confidence"], 0.9)

    def test_confusion_logic_prefers_fox_over_fos_with_strength_thirty(self) -> None:
        prediction = predict_disc_class_ocr(make_label_crop("FOX", strength=30, angle=45))
        self.assertEqual(prediction["code"], "FOX")
        self.assertNotIn("FOS", prediction["candidates"][:1])

    def test_confusion_logic_maps_cif_to_cip(self) -> None:
        prediction = predict_disc_class_ocr(make_label_crop("CIP", strength=5, angle=15, blur_sigma=1.2))
        self.assertEqual(prediction["code"], "CIP")

    def test_rotated_disc_text_is_read_for_ipm(self) -> None:
        prediction = predict_disc_class_ocr(make_label_crop("IPM", strength=10, angle=195, blur_sigma=0.9))
        self.assertEqual(prediction["code"], "IPM")
        self.assertIn(prediction["confidence_tier"], {"high_confidence_exact", "probable_match"})

    def test_blurred_text_requires_confirmation_instead_of_wrong_label(self) -> None:
        prediction = predict_disc_class_ocr(make_label_crop("FOX", strength=30, angle=45, blur_sigma=3.2))
        self.assertNotEqual(prediction["code"], "FOS")
        self.assertIn(
            prediction["confidence_tier"],
            {"high_confidence_exact", "probable_match", "uncertain_manual_confirmation_required"},
        )
        if prediction["code"] == "UNKNOWN":
            self.assertEqual(prediction["confidence_tier"], "uncertain_manual_confirmation_required")

    def test_prediction_is_deterministic_for_same_crop(self) -> None:
        crop = make_label_crop("SAM", strength=20, angle=180, blur_sigma=1.1)
        first = predict_disc_class_ocr(crop)
        second = predict_disc_class_ocr(crop)
        self.assertEqual(first["code"], second["code"])
        self.assertEqual(first["confidence_tier"], second["confidence_tier"])
        self.assertEqual(first["candidates"][:3], second["candidates"][:3])

    def test_validation_logic_requires_manual_review(self) -> None:
        validation = validate_results(
            [
                DiscMeasurement(
                    disc_id="disc-1",
                    index=1,
                    center=Point(x=10, y=10),
                    disc_radius_px=20,
                    disc_diameter_px=40,
                    detected_code="LZD",
                    final_code="LZD",
                    auto_diameter_px=120,
                    auto_diameter_mm=18,
                    final_diameter_mm=18,
                    measurement_confidence=0.5,
                    overall_confidence=0.4,
                    review_required=True,
                    status="review_required",
                    warnings=["Low confidence"],
                )
            ]
        )
        self.assertEqual(validation["status"], "REQUIRES_MANUAL_REVIEW")


class ZoneMeasurementTests(unittest.TestCase):
    def test_zone_measurement_detects_visible_zone(self) -> None:
        image = make_plate_image([DiscSpec(center=(250, 280), zone_radius_px=70)], image_size=600, plate_radius=250)
        result = measure_zone(image, (250.0, 280.0, 22.0))
        self.assertFalse(result["no_zone_fallback_used"])
        self.assertGreater(result["diameter_px"], 44.0)
        self.assertGreater(result["confidence"], 0.6)

    def test_zone_measurement_applies_no_zone_fallback(self) -> None:
        image = make_plate_image([DiscSpec(center=(250, 280), zone_radius_px=0)], image_size=600, plate_radius=250)
        result = measure_zone(image, (250.0, 280.0, 22.0))
        self.assertTrue(result["no_zone_fallback_used"])
        self.assertAlmostEqual(result["diameter_px"], 44.0, delta=4.0)


class ApiIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def _analyze_fixture(self):
        image = make_plate_image(
            [
                DiscSpec(center=(250, 280), zone_radius_px=70),
                DiscSpec(center=(560, 300), zone_radius_px=0),
                DiscSpec(center=(400, 560), zone_radius_px=48),
            ]
        )
        response = self.client.post(
            "/api/analyze",
            files={"image": ("synthetic.png", encode_png(image), "image/png")},
            data={"include_debug_artifacts": "true"},
        )
        self.assertEqual(response.status_code, 200, msg=response.text)
        return response.json()

    def test_analyze_endpoint_returns_reviewable_session(self) -> None:
        payload = self._analyze_fixture()
        self.assertIn("analysis_id", payload)
        self.assertEqual(len(payload["results"]), 3)
        self.assertIn("plate_overlay_base64", payload["debug_artifacts"])

    def test_review_endpoint_persists_manual_corrections(self) -> None:
        payload = self._analyze_fixture()
        analysis_id = payload["analysis_id"]
        first_disc = payload["results"][0]
        review_response = self.client.post(
            f"/api/analysis/{analysis_id}/review",
            json={
                "operator_id": "qa-user",
                "discs": [
                    {
                        "disc_id": first_disc["disc_id"],
                        "corrected_code": "LZD",
                        "corrected_diameter_mm": 31.5,
                        "operator_note": "Confirmed against fixture.",
                        "confirmed": True,
                    }
                ],
            },
        )
        self.assertEqual(review_response.status_code, 200, msg=review_response.text)
        updated = review_response.json()
        updated_disc = next(item for item in updated["results"] if item["disc_id"] == first_disc["disc_id"])
        self.assertEqual(updated_disc["final_code"], "LZD")
        self.assertEqual(updated_disc["final_diameter_mm"], 31.5)
        self.assertEqual(updated_disc["status"], "corrected")

    def test_export_endpoint_returns_table_columns(self) -> None:
        payload = self._analyze_fixture()
        analysis_id = payload["analysis_id"]
        export_response = self.client.get(f"/api/analysis/{analysis_id}/export")
        self.assertEqual(export_response.status_code, 200)
        self.assertIn("disc_id,code,auto_mm,corrected_mm,final_mm", export_response.text)


class RegressionFixtureTests(unittest.TestCase):
    @staticmethod
    def _assign_expected_latest_codes(results):
        expected_positions = {
            "FOX": (0.36, 0.30),
            "LZD": (0.65, 0.29),
            "TE": (0.28, 0.57),
            "CN": (0.78, 0.57),
            "CIP": (0.56, 0.78),
        }
        height = max(item.center.y for item in results)
        width = max(item.center.x for item in results)
        assigned = {}
        remaining = list(results)
        for code, (expected_x, expected_y) in expected_positions.items():
            best = min(
                remaining,
                key=lambda item: ((item.center.x / width) - expected_x) ** 2 + ((item.center.y / height) - expected_y) ** 2,
            )
            assigned[code] = best
            remaining.remove(best)
        return assigned

    def test_benchmark_fixture_tracks_lzd_above_thirty_mm(self) -> None:
        label_prediction = predict_disc_class_ocr(make_label_crop("LZD"))
        self.assertEqual(label_prediction["code"], "LZD")

        calibration = build_calibration([(0.0, 0.0, 30.0), (1.0, 1.0, 30.0), (2.0, 2.0, 30.0)])
        lzd_mm = calculate_inhibition_result(zone_diameter_px=320.0, disc_diameter_px=60.0, mm_per_pixel=calibration.mm_per_pixel)
        cip_mm = calculate_inhibition_result(zone_diameter_px=210.0, disc_diameter_px=60.0, mm_per_pixel=calibration.mm_per_pixel)
        amp_mm = calculate_inhibition_result(zone_diameter_px=180.0, disc_diameter_px=60.0, mm_per_pixel=calibration.mm_per_pixel)

        self.assertGreater(lzd_mm, 30.0)
        self.assertLess(cip_mm, 30.0)
        self.assertLess(amp_mm, 30.0)

    @unittest.skipUnless(REAL_SAMPLE_PATH.exists(), "Real client sample fixture is not available in this workspace.")
    def test_real_sample_plate_keeps_lzd_above_thirty_mm(self) -> None:
        result = _analyze_fixture(str(REAL_SAMPLE_PATH), "sample.jpeg")
        rows = {item.final_code: item.final_diameter_mm for item in result.results}
        self.assertIn("LZD", rows)
        self.assertGreater(rows["LZD"], 29.0)
        for code, diameter in rows.items():
            if code == "LZD":
                continue
            self.assertLess(diameter, 30.0, msg=f"{code} unexpectedly measured {diameter} mm")

    @unittest.skipUnless(LATEST_SAMPLE_PATH.exists(), "Latest uploaded sample fixture is not available in this workspace.")
    def test_latest_uploaded_sample_reads_visible_fox_not_fos(self) -> None:
        prediction = predict_disc_class_ocr(_load_latest_sample_fox_crop())
        self.assertEqual(prediction["code"], "FOX")
        self.assertNotEqual(prediction["code"], "FOS")

    @unittest.skipUnless(LATEST_SAMPLE_PATH.exists(), "Latest uploaded sample fixture is not available in this workspace.")
    def test_latest_uploaded_sample_pipeline_recovers_all_expected_discs(self) -> None:
        result = _analyze_fixture(str(LATEST_SAMPLE_PATH), "test1.jpeg")
        self.assertEqual(len(result.results), 5)
        assigned = self._assign_expected_latest_codes(result.results)
        self.assertEqual({code: disc.final_code for code, disc in assigned.items()}, {
            "FOX": "FOX",
            "LZD": "LZD",
            "TE": "TE",
            "CN": "CN",
            "CIP": "CIP",
        })
        self.assertTrue(all(not item.no_zone_fallback_used for item in assigned.values()))

    @unittest.skipUnless(LATEST_SAMPLE_PATH.exists(), "Latest uploaded sample fixture is not available in this workspace.")
    def test_latest_uploaded_sample_pipeline_is_deterministic(self) -> None:
        first = _analyze_fixture(str(LATEST_SAMPLE_PATH), "test1.jpeg")
        second = _analyze_fixture(str(LATEST_SAMPLE_PATH), "test1.jpeg")
        first_rows = [(item.final_code, round(item.final_diameter_mm, 2)) for item in sorted(first.results, key=lambda item: (item.center.x + item.center.y))]
        second_rows = [(item.final_code, round(item.final_diameter_mm, 2)) for item in sorted(second.results, key=lambda item: (item.center.x + item.center.y))]
        self.assertEqual(first_rows, second_rows)


if __name__ == "__main__":
    unittest.main()
