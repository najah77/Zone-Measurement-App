# Handover Notes

## What Is Being Handed Over
- A Flutter frontend for upload, review, and result presentation
- A FastAPI backend for plate analysis, persistence, and export
- Filesystem-backed session storage under `STORAGE_ROOT`
- A documentation set for overview, architecture, API, deployment, troubleshooting, and client testing

## Operational Responsibilities

### Frontend Owner
- build and publish the Flutter web client
- keep the backend base URL aligned with the deployed API
- update `antibiotic_rules.json` if interpretation coverage needs to grow

### Backend Owner
- run the FastAPI service
- manage Python dependencies and Tesseract installation
- maintain `STORAGE_ROOT` permissions, capacity, and backups
- monitor API availability and log files

## Day-To-Day Operator Expectations
- upload or capture a plate image
- review OCR suggestions and zone overlays
- correct any uncertain rows
- save the reviewed session
- use the result table and CSV export as the reviewed output

## Data That Must Be Preserved
The following directory is the operational record of the system:
- `STORAGE_ROOT/<analysis_id>/`

Each session directory may contain:
- `original_upload.bin`
- `analysis.json`
- debug overlays such as `plate_overlay.png`

Do not clear this directory without a retention policy, because it contains both the source image and the saved review state.

## Backup Recommendation
- Back up `STORAGE_ROOT` daily or according to client policy
- Back up the deployed frontend build artifact for rollback
- Back up the backend environment file or secret configuration separately from the repository

## Known Limitations At Handover
- No database-backed audit trail or authentication layer is present
- Export is CSV only
- Frontend interpretation thresholds cover only the codes listed in `frontend/lib/data/local/antibiotic_rules.json`
- Native Android and iOS release packaging still need platform-specific release setup
- There is no built-in admin panel for browsing historical analyses

## Safe Update Practice
1. Back up `STORAGE_ROOT`.
2. Run backend tests.
3. Run the Flutter test.
4. Deploy backend and frontend together when response fields change.
5. Validate one real image end to end after deployment.

## Immediate Items To Address Before Client Go-Live
- Rotate any real secrets currently present in local `.env` files.
- Decide whether client delivery will be:
  - Flutter web plus hosted backend
  - or a later native mobile release after platform hardening
- Confirm the production storage path and backup policy.
- Expand `antibiotic_rules.json` if the client expects in-app interpretation labels for a broader antibiotic panel.
