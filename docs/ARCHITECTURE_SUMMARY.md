# Architecture Summary

## Frontend
- Flutter app with screen-based flow:
  - Home
  - Image upload or capture
  - Analysis review
  - Final results table
- `AnalysisViewModel` manages upload analysis, operator edits, and review save.
- `ResultViewModel` converts saved backend results into the final operator table.

## Backend
- FastAPI service with four main endpoints:
  - analyze upload
  - fetch analysis by id
  - save manual review
  - export CSV
- `ast_processor.py` orchestrates upload validation, plate extraction, quality checks, analysis, and persistence.
- `inference.py` performs disc detection, OCR suggestion, calibration, zone measurement, confidence scoring, and debug artifact generation.

## Persistence
- File-based JSON record per analysis under `Backend/data/analysis_runs/<analysis_id>/analysis.json`
- Original uploaded image stored beside the JSON record for traceability.

## Review Model
- Automatic output is preserved.
- Manual edits are stored separately as corrected code and corrected diameter.
- Final output is derived without overwriting the original automatic values.
