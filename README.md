# Zone Measurement App

Production-oriented AST zone measurement application for antibiotic disc diffusion plate reading.

## Stack
- Frontend: Flutter
- Backend: FastAPI + OpenCV + local OCR
- Persistence: JSON records stored under `Backend/data/analysis_runs`

## Quick Start
1. Start the backend:
   - `cd Backend`
   - `venv\Scripts\python.exe -m uvicorn app.main:app --reload`
2. Start the Flutter app:
   - `cd frontend`
   - `flutter run`

## Key Features
- Plate crop and quality checks
- Automatic disc detection and 6 mm calibration
- Automatic label suggestion with confidence-based confirmation
- Automatic zone measurement with 6 mm no-zone fallback
- Per-disc manual correction with saved review history
- Structured final table plus CSV export endpoint

## API
- `POST /api/analyze`
- `GET /api/analysis/{analysis_id}`
- `POST /api/analysis/{analysis_id}/review`
- `GET /api/analysis/{analysis_id}/export`

## Docs
- [Local Setup](docs/LOCAL_SETUP.md)
- [Production Deployment](docs/PRODUCTION_DEPLOYMENT.md)
- [Architecture Summary](docs/ARCHITECTURE_SUMMARY.md)
- [Measurement Pipeline](docs/MEASUREMENT_PIPELINE.md)
- [Operator Guide](docs/OPERATOR_GUIDE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Known Limitations](docs/KNOWN_LIMITATIONS.md)
- [QA Checklist](docs/QA_CHECKLIST.md)
- [Handover Notes](docs/HANDOVER_NOTES.md)
