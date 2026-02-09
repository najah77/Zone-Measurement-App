# AST Analyzer Backend (Phase 3)

FastAPI backend for Antibiotic Sensitivity Test (AST) Analyzer.

## Tech Stack
- Python 3.10+
- FastAPI
- Uvicorn
- OpenCV (opencv-python)
- NumPy
- Pydantic

## Setup

1.  **Create and Activate Virtual Environment (Recommended):**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the server:**
    ```bash
    uvicorn app.main:app --reload
    ```

3. Access API docs:
   - Swagger UI: http://127.0.0.1:8000/docs
   - ReDoc: http://127.0.0.1:8000/redoc

## API Endpoints

### POST /analyze
Accepts an image file (multipart/form-data) and returns detected antibiotic discs and zone diameters.

**Request:**
- `image`: File (JPEG/PNG)

**Response:**
```json
{
  "results": [
    { "code": "AMP", "diameter": 18.5 },
    { "code": "CIP", "diameter": 14.2 }
  ]
}
```
