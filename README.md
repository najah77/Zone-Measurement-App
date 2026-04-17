# Zone Measurement App

This repository contains a production-oriented antibiotic disc diffusion plate reading application. The app captures or uploads a plate image, detects the Petri dish and antibiotic discs, reads the disc code, measures inhibition zones in millimeters, guides the operator through manual confirmation when confidence is low, and stores the reviewed result set under a persistent `analysis_id`.

The current implementation is a Flutter client backed by a FastAPI image-analysis service. Automatic measurement is available, but the product is intentionally review-safe: uncertain labels or weak zone boundaries are surfaced for confirmation instead of being silently treated as certain.

## Confirmed Stack
- Frontend: Flutter `3.16.9` with Dart `3.2.6`
- Frontend dependencies: `provider`, `http`, `http_parser`, `image_picker`, `google_fonts`
- Backend runtime: Python `3.12.4`
- Backend framework: FastAPI `0.128.4` with Uvicorn `0.40.0`
- Image processing: OpenCV `4.13.0.92`, NumPy, SciPy, scikit-image
- OCR: local Tesseract OCR through `pytesseract`
- Persistence: filesystem-backed JSON sessions under `Backend/data/analysis_runs`
- Test frameworks: Python `unittest`, FastAPI `TestClient`, Flutter `flutter_test`

## Repository Structure
- `Backend/`: FastAPI service, image-analysis pipeline, tests, storage
- `frontend/`: Flutter application for upload, review, and results
- `docs/`: technical, deployment, troubleshooting, client testing, and handover documentation

## Quick Start
1. Start the backend:
   ```powershell
   cd C:\Users\najah\Desktop\Zone_Measurement_App\Backend
   venv\Scripts\python.exe -m uvicorn app.main:app --reload
   ```
2. Start the Flutter app:
   ```powershell
   cd C:\Users\najah\Desktop\Zone_Measurement_App\frontend
   flutter pub get
   flutter run --dart-define=BACKEND_BASE_URL=http://127.0.0.1:8000
   ```
3. Open the API docs:
   - [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Documentation
- [Application Overview](docs/APP_OVERVIEW.md)
- [Architecture](docs/ARCHITECTURE.md)
- [API Reference](docs/API_REFERENCE.md)
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)
- [Client Testing Guide](docs/CLIENT_TESTING_GUIDE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Handover Notes](docs/HANDOVER_NOTES.md)

## Important Deployment Notes
- The backend requires a writable persistent directory for `STORAGE_ROOT`. Without persistent storage, `analysis_id` records, review edits, and exports will not survive restarts.
- The backend OCR path uses local Tesseract. Tesseract must be installed on the host and available on `PATH`.
- The Flutter client now supports a deployment-safe backend URL override through `--dart-define=BACKEND_BASE_URL=...`.
- The FastAPI backend now supports deployment-safe CORS configuration through `CORS_ALLOW_ORIGINS`.
- The repo currently contains legacy docs and a local `.env`. Treat any locally stored real secrets as compromised until rotated and moved into deployment secret storage.
