# Backend

FastAPI backend for AST plate analysis.

## Run
- `venv\Scripts\python.exe -m uvicorn app.main:app --reload`

## Endpoints
- `POST /api/analyze`
- `GET /api/analysis/{analysis_id}`
- `POST /api/analysis/{analysis_id}/review`
- `GET /api/analysis/{analysis_id}/export`

## Storage
- Analysis records are stored under `Backend/data/analysis_runs`

## Tests
- `venv\Scripts\python.exe -m unittest discover -s tests -v`
