# Troubleshooting

## Backend Does Not Start

### Symptom
`uvicorn` exits immediately or the app fails on import.

### Checks
- Confirm the virtual environment exists and dependencies are installed:
  ```powershell
  cd C:\Users\najah\Desktop\Zone_Measurement_App\Backend
  venv\Scripts\python.exe -m pip install -r requirements.txt
  ```
- Verify the app imports:
  ```powershell
  venv\Scripts\python.exe -m compileall app
  ```

## Tesseract OCR Not Found

### Symptom
OCR returns empty results or `pytesseract` raises an error about the Tesseract executable.

### Cause
The backend uses local Tesseract through `pytesseract`. Tesseract must be installed on the server and available on `PATH`.

### Fix
- Install Tesseract on the backend host.
- Restart the backend service.
- Re-run a test image and check that `raw_ocr_text` is no longer empty for readable discs.

## Frontend Cannot Reach Backend

### Symptom
The Flutter app loads, but analysis fails with a network or timeout error.

### Checks
- Confirm the frontend was built or run with the correct backend URL:
  ```bash
  flutter run --dart-define=BACKEND_BASE_URL=http://127.0.0.1:8000
  flutter build web --dart-define=BACKEND_BASE_URL=https://api.example.com
  ```
- Confirm the backend is reachable at:
  - `/`
  - `/docs`
  - `/api/openapi.json`

## Analysis Submit, Polling, Or Result Fetch Fails

### Symptom
The app submits an image, but analysis appears to stall, fail, or never reach the review screen.

### Actual Behavior In This Repo
Real image analysis can take around 1 to 3 minutes on the current backend pipeline, especially for high-resolution plate images.

### Current Frontend Behavior
- `POST /api/analyze` should return quickly with an `analysis_id`
- the Flutter client then polls `GET /api/analyze/{analysis_id}/status`
- once status reaches `completed`, the client fetches `GET /api/analyze/{analysis_id}/result`
- JSON parsing is offloaded from the main UI thread
- large preview images are decoded lazily instead of during widget build

### If The Flow Still Fails
- confirm the backend is not stalled
- confirm the frontend points to the correct `BACKEND_BASE_URL`
- confirm the installed APK is a build that includes the async submit/poll/result flow
- inspect `status.json` under `STORAGE_ROOT/<analysis_id>`
- remember that `GET /api/analyze/{analysis_id}/result` returns `409` until the job is complete

## Browser CORS Error

### Symptom
The browser blocks API requests from the deployed frontend.

### Cause
The backend now reads allowed origins from `CORS_ALLOW_ORIGINS`.

### Fix
- Set:
  ```env
  CORS_ALLOW_ORIGINS=https://app.example.com
  ```
- Restart the backend.

## Upload Rejected

### Symptom
`POST /api/analyze` returns `400` or `413`.

### Actual Backend Rules
- non-image files: `File must be an image.`
- empty image: `Uploaded image is empty.`
- oversize image: `Image exceeds the <n> MB upload limit.`

### Fix
- Upload a valid image file
- Increase `MAX_UPLOAD_BYTES` if your hosted environment needs a larger limit

## Proxy Or Cloudflare 524 Errors

### Symptom
The old mobile build waits on one long analyze request and receives `524` from Cloudflare or another reverse proxy.

### Current Behavior
The production-safe API path is now asynchronous. `POST /api/analyze` returns quickly, and the app polls status instead of waiting for OCR and measurement to finish inside one request.

### Fix
- restart the backend after deploying the async API changes
- rebuild and reinstall the Flutter app so it uses the new submit/poll/result flow
- verify the app is calling:
  - `POST /api/analyze`
  - `GET /api/analyze/{analysis_id}/status`
  - `GET /api/analyze/{analysis_id}/result`

## Analysis Saves Fail Or Records Disappear

### Symptom
The app analyzes an image but later fetch or export by `analysis_id` fails.

### Cause
The backend stores sessions on disk. If `STORAGE_ROOT` is unwritable or ephemeral, sessions will be lost.

### Fix
- Point `STORAGE_ROOT` to a persistent writable directory
- Verify the backend process user can create folders and files there
- Back up the storage path regularly

## Job Stays In `queued` Or `processing`

### Symptom
The client keeps polling, but the job never reaches `completed`.

### Current Safeguard
On backend startup, any orphaned `queued` or `processing` job is marked `failed` with the message `Analysis was interrupted before completion. Please retry the upload.`

### Fix
- check whether the backend restarted mid-analysis
- confirm the backend can write to `STORAGE_ROOT`
- if the job is now `failed`, re-upload the image and start a new analysis

## OCR Is Weak Or Requires Manual Confirmation

### Symptom
Some discs return `probable_match`, `uncertain_manual_confirmation_required`, or `failed_unknown`.

### Explanation
This is expected safety behavior when OCR evidence is weak. The current backend prefers review-required output over a wrong confident label.

### Operator Guidance
- verify the visible label manually
- use the fallback selector when needed
- save the reviewed correction so the final session reflects the confirmed value

## Large Visible Zone Falls Back To `6 mm`

### Symptom
The row shows `no_zone_fallback_used=true` even though the zone is visible.

### Checks
- Inspect image quality warnings for blur, glare, clipping, or low contrast
- Check the per-disc overlay in the review screen
- Inspect `analysis.json` and any overlay images under `STORAGE_ROOT/<analysis_id>`

### Notes
The current zone logic uses multiple signals before falling back to `6 mm`, but weak images can still push rows into review-required states. In those cases, use manual correction rather than accepting the fallback blindly.

## Result Screen Shows `Unknown (CODE)`

### Cause
The Flutter result screen maps antibiotic names and interpretation thresholds from `frontend/lib/data/local/antibiotic_rules.json`.

### Current Limitation
That file currently covers only a limited subset of antibiotic codes. If a detected code is missing from that local asset, the result screen will still show the code but name and interpretation remain unknown.

## Native Mobile Release Issues

### Android
- `android/app/build.gradle` still uses placeholder-style release configuration
- production signing and final application ID still need to be set before store delivery

### iOS
- `ios/Runner/Info.plist` still needs camera and photo-library usage descriptions before production distribution

## Windows Camera Capture

### Symptom
Windows desktop shows an ImagePicker camera delegate error.

### Current Behavior
The frontend now treats camera capture as unsupported on Windows desktop and hides the broken camera action. Use file upload instead.

## Useful Verification Commands
```powershell
cd C:\Users\najah\Desktop\Zone_Measurement_App
Backend\venv\Scripts\python.exe -m unittest C:\Users\najah\Desktop\Zone_Measurement_App\Backend\tests\test_analysis_pipeline.py -v
Backend\venv\Scripts\python.exe -m compileall C:\Users\najah\Desktop\Zone_Measurement_App\Backend\app
cd frontend
flutter test test/analysis_session_model_test.dart
```
