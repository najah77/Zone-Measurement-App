# Production Deployment

## Backend
- Serve the FastAPI app behind a reverse proxy such as Nginx or Traefik.
- Run a production ASGI server command such as:
  - `venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2`
- Persist `Backend/data/analysis_runs` on durable storage.
- Set environment variables for:
  - `APP_VERSION`
  - `ALGORITHM_VERSION`
  - `MAX_UPLOAD_BYTES`
  - `STORAGE_ROOT`
  - `OCR_MIN_CONFIDENCE`
  - `MEASUREMENT_MIN_CONFIDENCE`

## Frontend
- Build a release bundle with the platform-specific Flutter command.
- Point the frontend API base URL at the deployed FastAPI host before release packaging.
- Validate camera permissions for Android, iOS, and desktop targets used by the client.

## Operational Notes
- Keep a backup policy for the analysis run directory.
- Monitor upload failures, review-required rate, and storage growth.
- Rotate release notes whenever the algorithm version changes.
