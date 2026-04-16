# Local Setup

## Backend
1. Open a terminal in `Backend`.
2. Create or reuse the existing virtual environment:
   - `python -m venv venv`
   - `venv\Scripts\activate`
3. Install dependencies:
   - `pip install -r requirements.txt`
4. Run the API:
   - `venv\Scripts\python.exe -m uvicorn app.main:app --reload`

## Frontend
1. Open a second terminal in `frontend`.
2. Install Flutter packages:
   - `flutter pub get`
3. Run the app:
   - `flutter run`

## Test Commands
- Backend:
  - `Backend\venv\Scripts\python.exe -m unittest discover -s Backend\tests -v`
- Frontend formatting:
  - `C:\dart-sdk\bin\dart.exe format lib`

## Default Local URLs
- Backend docs: `http://127.0.0.1:8000/docs`
- Backend API root: `http://127.0.0.1:8000/`
