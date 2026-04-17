# API Reference

## Base URL
- Local development: `http://127.0.0.1:8000`
- Swagger UI: `/docs`
- OpenAPI JSON: `/api/openapi.json`

All application endpoints are mounted under `/api`.

## Endpoint Summary

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/analyze` | Upload an image and create a new analysis session |
| `GET` | `/api/analysis/{analysis_id}` | Fetch a previously saved analysis session |
| `POST` | `/api/analysis/{analysis_id}/review` | Save operator corrections for one or more discs |
| `GET` | `/api/analysis/{analysis_id}/export` | Export the saved result table as CSV |
| `GET` | `/` | Backend health-style root message |

## `POST /api/analyze`

### Purpose
Accepts an uploaded image, runs plate analysis, persists the session, and returns the complete analysis payload.

### Request
- Content type: `multipart/form-data`
- Form fields:
  - `image`: required uploaded image file
  - `include_debug_artifacts`: optional boolean form field

### Validation
- Rejects files whose content type does not start with `image/`
- Rejects empty uploads
- Rejects uploads larger than `MAX_UPLOAD_BYTES`

### Example cURL
```bash
curl -X POST "http://127.0.0.1:8000/api/analyze" \
  -F "image=@sample_plate.jpeg" \
  -F "include_debug_artifacts=true"
```

### Response Shape
```json
{
  "analysis_id": "6d8f5e8d-93ff-4a4e-b75c-8a93ad8f1d92",
  "status": "REQUIRES_MANUAL_REVIEW",
  "algorithm_version": "zone-measurement-2.0.0",
  "created_at": "2026-04-18T09:11:22.947755+00:00",
  "image_filename": "sample_plate.jpeg",
  "plate_detection": {
    "center": {"x": 622.5, "y": 618.4},
    "radius_px": 520.8,
    "clipped": false,
    "clip_fraction": 0.0,
    "method": "hough"
  },
  "quality_report": {
    "blur_score": 74.22,
    "brightness": 167.95,
    "contrast": 25.87,
    "glare_fraction": 0.0041,
    "review_required": false,
    "warnings": []
  },
  "calibration": {
    "disc_diameter_mm": 6.0,
    "average_disc_diameter_px": 44.0,
    "mm_per_pixel": 0.1363,
    "disc_count": 5,
    "spread_px": 1.5,
    "method": "median-disc-diameter"
  },
  "summary": {
    "total_discs": 5,
    "review_required_count": 2,
    "corrected_count": 0,
    "failed_count": 0,
    "auto_count": 3
  },
  "warnings": [],
  "debug_artifacts": {
    "plate_overlay_base64": "..."
  },
  "results": [
    {
      "disc_id": "disc-1",
      "index": 1,
      "detected_code": "FOX",
      "final_code": "FOX",
      "label_confidence": 0.94,
      "label_confidence_tier": "probable_match",
      "label_candidates": ["FOX", "FOS"],
      "whitelist_candidates_considered": ["FOX", "FOS", "CN"],
      "label_engine": "hybrid-ocr-whitelist",
      "label_decision_source": "ocr+whitelist_correction",
      "label_selection_reason": "Rotation voting and whitelist scoring favored 'FOX' over nearby alternatives; manual confirmation is still required.",
      "raw_ocr_text": "FO | 30",
      "normalized_ocr_text": "FO 30",
      "layout_suggestion": null,
      "auto_diameter_px": 196.2,
      "auto_diameter_mm": 26.76,
      "corrected_diameter_mm": null,
      "final_diameter_mm": 26.76,
      "measurement_confidence": 0.81,
      "overall_confidence": 0.88,
      "source": "auto",
      "status": "review_required",
      "review_required": true,
      "no_zone_fallback_used": false,
      "warnings": [],
      "measurement_method": "hybrid-zone-boundary",
      "calibration_mm_per_pixel": 0.1363,
      "crop_image_base64": "...",
      "overlay_image_base64": "..."
    }
  ]
}
```

### Common Errors
- `400`: `File must be an image.`
- `400`: `Uploaded image is empty.`
- `400`: image decode error from `load_image_from_bytes`
- `413`: `Image exceeds the <n> MB upload limit.`

## `GET /api/analysis/{analysis_id}`

### Purpose
Returns the persisted analysis payload for an existing `analysis_id`.

### Example
```bash
curl "http://127.0.0.1:8000/api/analysis/6d8f5e8d-93ff-4a4e-b75c-8a93ad8f1d92"
```

### Response
- Same `AnalysisResponse` structure returned by `POST /api/analyze`

### Errors
- `404`: `Analysis record not found`

## `POST /api/analysis/{analysis_id}/review`

### Purpose
Persists operator confirmation and corrections for any subset of detected discs.

### Request Body
```json
{
  "operator_id": "qa-user",
  "discs": [
    {
      "disc_id": "disc-1",
      "corrected_code": "FOX",
      "corrected_diameter_mm": 27.5,
      "operator_note": "Boundary adjusted after visual check.",
      "confirmed": true
    }
  ]
}
```

### Review Rules
- `corrected_diameter_mm` is normalized by backend logic and cannot end below `6 mm`
- `corrected_code` is uppercased before persistence
- `confirmed=true` clears `review_required` when the row is still automatic
- Corrected rows are marked `source=manual` and `status=corrected`

### Response
- Returns the updated `AnalysisResponse`

### Errors
- `400`: `Unknown disc id: <disc_id>`
- `404`: `Analysis record not found`

## `GET /api/analysis/{analysis_id}/export`

### Purpose
Returns a CSV export generated from the persisted review state.

### Response
- Content type: `text/csv`
- Header:
```text
row,disc_id,code,auto_mm,corrected_mm,final_mm,label_confidence,measurement_confidence,overall_confidence,status,source,review_required,no_zone_fallback_used,warnings
```

### Example
```bash
curl -OJ "http://127.0.0.1:8000/api/analysis/6d8f5e8d-93ff-4a4e-b75c-8a93ad8f1d92/export"
```

## `GET /`

### Purpose
Very light root endpoint used as a basic reachability check.

### Response
```json
{
  "message": "AST Analyzer Backend Running"
}
```

## Data Semantics

### Status Fields
- `status=VALID`: no disc remains in a failed or review-required state
- `status=REQUIRES_MANUAL_REVIEW`: one or more discs still require confirmation or failed automatic validation

### Per-Disc `status`
- `auto`: automatic value kept as final
- `corrected`: operator changed code and/or diameter
- `review_required`: backend needs operator confirmation
- `failed`: backend could not safely measure or classify the row

### Confidence And Safety Fields
- `label_confidence`: OCR/whitelist confidence for the selected label
- `measurement_confidence`: zone-measurement confidence
- `overall_confidence`: combined per-disc confidence
- `no_zone_fallback_used`: true only when the zone logic falls back to the 6 mm minimum
- `warnings`: operator-visible review notes for that disc

### Debug Artifacts
If `include_debug_artifacts=true` is sent to the analyze endpoint, the backend may return and persist:
- `plate_overlay_base64`
- disc crop overlays
- filesystem paths such as `plate_overlay_path` after persistence
