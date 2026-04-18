# API Reference

## Base URL
- Local development: `http://127.0.0.1:8000`
- Swagger UI: `/docs`
- OpenAPI JSON: `/api/openapi.json`

All application endpoints are mounted under `/api`.

## Endpoint Summary

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/analyze` | Upload an image, create an async analysis job, and return immediately with `analysis_id` |
| `GET` | `/api/analyze/{analysis_id}/status` | Poll the async job state |
| `GET` | `/api/analyze/{analysis_id}/result` | Fetch the final analysis payload after completion |
| `GET` | `/api/analysis/{analysis_id}` | Fetch a previously saved analysis session |
| `POST` | `/api/analysis/{analysis_id}/review` | Save operator corrections for one or more discs |
| `GET` | `/api/analysis/{analysis_id}/export` | Export the saved result table as CSV |
| `GET` | `/` | Backend health-style root message |

## `POST /api/analyze`

### Purpose
Accepts an uploaded image, creates a background analysis job, persists the upload, and returns immediately with an `analysis_id`.

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
  "status": "queued",
  "created_at": "2026-04-18T09:11:22.947755+00:00",
  "message": "Upload received. Analysis job queued.",
  "status_url": "/api/analyze/6d8f5e8d-93ff-4a4e-b75c-8a93ad8f1d92/status",
  "result_url": "/api/analyze/6d8f5e8d-93ff-4a4e-b75c-8a93ad8f1d92/result"
}
```

### Common Errors
- `400`: `File must be an image.`
- `400`: `Uploaded image is empty.`
- `400`: image decode error from `load_image_from_bytes`
- `413`: `Image exceeds the <n> MB upload limit.`

## `GET /api/analyze/{analysis_id}/status`

### Purpose
Returns the current async job state without re-running analysis.

### Status Values
- `queued`
- `processing`
- `completed`
- `failed`

If the backend restarts while a job is still `queued` or `processing`, startup recovery marks that job as `failed` with a retry message instead of leaving it stuck indefinitely.

### Example Response
```json
{
  "analysis_id": "6d8f5e8d-93ff-4a4e-b75c-8a93ad8f1d92",
  "status": "processing",
  "created_at": "2026-04-18T09:11:22.947755+00:00",
  "updated_at": "2026-04-18T09:12:03.128312+00:00",
  "image_filename": "sample_plate.jpeg",
  "message": "Processed disc 2 of 5.",
  "progress": 0.61,
  "current_stage": "disc_analysis",
  "error": null,
  "result_available": false,
  "timings": {
    "image_decode_seconds": 0.09,
    "plate_extraction_seconds": 3.12,
    "disc_detection_seconds": 3.67
  },
  "status_url": "/api/analyze/6d8f5e8d-93ff-4a4e-b75c-8a93ad8f1d92/status",
  "result_url": "/api/analyze/6d8f5e8d-93ff-4a4e-b75c-8a93ad8f1d92/result"
}
```

## `GET /api/analyze/{analysis_id}/result`

### Purpose
Returns the final `AnalysisResponse` after the async job has completed.

### Response
- `200`: full `AnalysisResponse`
- `409`: job is still `queued` or `processing`
- `409`: job is `failed`, with backend error detail

## `GET /api/analysis/{analysis_id}`

### Purpose
Returns the persisted analysis payload for a completed `analysis_id`. This remains as a compatibility endpoint for saved sessions.

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
