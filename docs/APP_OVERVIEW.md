# Application Overview

## Product Summary
The Zone Measurement App is an assisted reading system for antibiotic disc diffusion plates. It is designed for operators who need a faster, more repeatable workflow than a fully manual dashed-circle process, while still retaining safe confirmation controls when image quality, OCR, or zone boundaries are uncertain.

The current app accepts a captured or uploaded plate image, isolates the plate, detects discs, calibrates pixels to millimeters using the known 6 mm disc diameter, reads antibiotic codes from each disc, measures the surrounding inhibition zone, and presents an operator review screen before producing a final result table.

## Problem It Solves
- Replaces a mostly manual zone placement workflow with automatic plate, disc, OCR, and zone analysis.
- Preserves a review-safe manual correction path instead of forcing operators to trust uncertain automation.
- Keeps a persistent `analysis_id` session so a single analysis can be fetched, reviewed, corrected, and exported consistently.
- Produces a structured final table instead of free-form manual notes.

## Target Users
- Laboratory operators reading disc diffusion plates
- Supervisors reviewing borderline or uncertain automated readings
- Client UAT teams validating upload, measurement, review, and export workflows

## Core Features In The Current Repo
- Camera or gallery image selection in the Flutter app through `image_picker`
- Automatic plate detection with clipped-plate warnings
- Automatic disc detection with geometric filtering and Hough fallback
- Automatic calibration using the known 6 mm disc diameter
- Automatic OCR label suggestion with whitelist matching and confidence tiers
- Automatic inhibition zone measurement with edge, contour, and radial evidence
- Manual review controls for code confirmation and diameter correction
- Saved review state through `analysis_id`
- CSV export from the persisted analysis session

## End-To-End Workflow
1. The operator opens the Flutter app and captures or uploads an image.
2. The frontend posts the image to `POST /api/analyze`.
3. The backend validates the upload, decodes the image, extracts the plate, checks image quality, and runs the hybrid analysis pipeline.
4. The backend stores the original upload and the analysis JSON under a new `analysis_id`.
5. The frontend shows the analysis review screen with:
   - quality warnings
   - calibration summary
   - per-disc OCR suggestion
   - per-disc zone overlay
   - per-disc slider, step buttons, numeric correction, and fallback selector
6. The operator confirms or corrects codes and diameters.
7. The frontend posts the reviewed session to `POST /api/analysis/{analysis_id}/review`.
8. The backend persists the corrections without destroying the original automatic values.
9. The frontend shows the final result table with auto, corrected, and final values.
10. The backend can export the saved table through `GET /api/analysis/{analysis_id}/export`.

## Safety And Review Logic
- Final diameter is never allowed below `6 mm`.
- The known white-disc diameter of `6 mm` is the calibration reference for millimeter conversion.
- `6 mm` is also the minimum no-zone fallback value.
- The backend marks rows `review_required` when OCR confidence, measurement confidence, image quality, or geometry checks are not strong enough.
- The frontend keeps manual selection and manual diameter adjustment available as a fallback, not as the primary path.
- Low-confidence OCR should surface as confirmation-needed rather than a wrong high-confidence label.

## Recommended Usage
- Use a well-lit, centered plate image with minimal glare.
- Review every row that shows `review_required`, `probable_match`, or warnings about blur, glare, low contrast, or clipped edges.
- Use the slider or numeric control when the zone overlay is visibly too small or too large.
- Treat the CSV export as the reviewed output only after review-required rows have been confirmed.

## Known Limitations In This Repository
- Persistence is filesystem-based only. There is no database, user management layer, or server-side audit trail beyond the stored JSON and optional `operator_id`.
- The result interpretation names in the Flutter app rely on `frontend/lib/data/local/antibiotic_rules.json`, which currently contains a limited set of antibiotic rules rather than a full lab panel.
- The current Flutter UI uses analyze and review endpoints directly. The backend fetch and export endpoints exist and are production-usable, but export is not exposed as a dedicated in-app download action today.
- OCR is local Tesseract OCR with whitelist correction. It is stronger than the original manual workflow, but faint, rotated, or glare-obscured text can still require manual confirmation.
- Native mobile release packaging is not fully handover-ready for app-store distribution. The easiest deployment path in the current repo is Flutter web plus FastAPI hosting.
