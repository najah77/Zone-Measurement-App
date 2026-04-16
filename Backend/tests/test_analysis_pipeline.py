from __future__ import annotations

import sys
import unittest
from pathlib import Path

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
from app.services.expert_engine import validate_results
from app.utils.calibration import build_calibration, pixels_to_mm
from app.utils.disc_detector import validate_disc_geometry
from app.utils.measurement import apply_no_zone_rule, calculate_inhibition_result
from app.utils.zone_detector import measure_zone
from synthetic_plate import DiscSpec, encode_png, make_label_crop, make_plate_image


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


if __name__ == "__main__":
    unittest.main()
