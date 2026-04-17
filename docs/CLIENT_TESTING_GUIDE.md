# Client Testing Guide

## Purpose
This guide is for client UAT and operator acceptance testing on the deployed system. It focuses on the workflows that exist in the current repository: upload, automatic analysis, manual review, save, fetch by `analysis_id`, and CSV export.

## URLs To Provide To The Client
- Frontend URL: `https://app.example.com`
- Backend API docs URL: `https://api.example.com/docs`
- Optional direct export pattern:
  - `https://api.example.com/api/analysis/{analysis_id}/export`

## Pre-Test Checklist
- The frontend loads without browser console API errors.
- The backend `/docs` page is reachable.
- The backend storage directory is writable.
- A sample test image is available on the tester device.

## What The Client Should Test

### 1. Image Upload And Start Analysis
1. Open the frontend URL.
2. Choose camera or gallery upload.
3. Select a real plate image.
4. Confirm the analysis screen appears instead of a crash or blank page.

Expected result:
- the app displays the image preview or plate overlay
- disc cards are listed
- quality warnings appear if the image is blurry, dark, clipped, or glared

### 2. Automatic Detection Review
For each disc card, confirm:
- a suggested antibiotic code is shown
- the confidence state is visible
- the measured diameter in millimeters is visible
- the overlay or crop preview is visible
- warnings appear on uncertain rows

Expected result:
- strong rows stay automatic
- weaker rows are clearly flagged for confirmation instead of silently hidden

### 3. Manual Code Confirmation
1. Open a disc whose label needs confirmation.
2. Use the dropdown fallback if the visible label does not match the automatic suggestion.
3. Save the corrected code.

Expected result:
- the row is marked corrected after review save
- the final code persists

### 4. Manual Diameter Correction
1. Pick a disc whose zone boundary needs adjustment.
2. Use the slider, plus/minus controls, or numeric input to change the diameter.
3. Save the review.

Expected result:
- `final_diameter_mm` reflects the corrected value
- the corrected value never goes below `6 mm`

### 5. Result Table Review
After saving, confirm the results screen shows:
- row number
- antibiotic code
- auto diameter
- corrected diameter where applicable
- final diameter
- status
- warnings
- confidence-based review indicators

### 6. analysis_id Verification
Ask the deployment team to retrieve the same session by `analysis_id` using:
```bash
curl "https://api.example.com/api/analysis/<analysis_id>"
```

Expected result:
- the JSON matches the reviewed session shown in the app

### 7. CSV Export
Ask the deployment team to open:
```text
https://api.example.com/api/analysis/<analysis_id>/export
```

Expected result:
- a CSV downloads
- the CSV includes code, auto value, corrected value, final value, status, source, and warnings

## Suggested UAT Scenarios
- Well-lit plate with clearly visible labels
- Plate with mild blur
- Plate with one edge-near disc
- Plate with weak zone boundary
- Plate where at least one label requires manual confirmation

## What Counts As A Pass
- The app stays stable through upload, review, save, and export
- `analysis_id` remains consistent across fetch and export
- corrected values persist after save
- rows that are uncertain are clearly marked for review
- no final diameter drops below `6 mm`

## What To Report If A Problem Is Found
Provide:
- the image used
- the `analysis_id`
- the incorrect disc code or diameter
- whether the issue was automatic detection, save, fetch, or export
- a screenshot of the review screen if available

## Where The Deployment Team Can Inspect Server Artifacts
On the backend host, each run is stored under:
- `STORAGE_ROOT/<analysis_id>/original_upload.bin`
- `STORAGE_ROOT/<analysis_id>/analysis.json`
- optional debug images such as `plate_overlay.png`

Those files are the fastest way to inspect whether the backend persisted the same session the client saw in the UI.
