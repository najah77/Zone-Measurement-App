# Deployment Guide

## Recommended Deployment Strategy

### Best Fit For This Repository
The most practical deployment for the current codebase is:
- Flutter web frontend
- FastAPI backend on a VM or server with Python `3.12.x`
- local Tesseract OCR installed on the backend host
- persistent filesystem storage mounted for `STORAGE_ROOT`
- reverse proxy or separate subdomain setup for frontend and backend

This recommendation is based on the real implementation:
- the backend writes review state, job status, and original uploads to disk
- the frontend can now be built against a configurable backend base URL
- the analyze path is now async-job based, so mobile traffic no longer depends on one long blocking HTTP response
- reverse proxies such as Cloudflare no longer need to wait for OCR and zone measurement to finish inside a single request
- there is no Docker, container orchestration, or database layer in the repo today
- native mobile packaging still needs platform-specific production hardening

### Low-Cost Client UAT Path
- One Linux VM
- Nginx serving the Flutter web build
- Uvicorn running the FastAPI backend behind Nginx
- Persistent disk mounted for uploads and analysis JSON
- Two subdomains:
  - `https://app.example.com` for the frontend
  - `https://api.example.com` for the backend

### Production Path
- Same architecture as UAT, but with:
  - managed TLS
  - backups for `STORAGE_ROOT`
  - process supervision such as `systemd`
  - access logging and rotation
  - release-specific Flutter build and backend env files

### Not Recommended Without Changes
- Stateless PaaS hosting with ephemeral disks
- Native mobile store release from the current repo without adding proper release signing and platform permissions

## Prerequisites

### Backend Host
- Python `3.12.x`
- `pip`
- Tesseract OCR installed and available on `PATH`
- Writable persistent directory for analysis storage

### Frontend Build Host
- Flutter `3.16.9` or a compatible `3.16.x` stable toolchain
- Dart `3.2.6`

## Environment Variables

### Backend Variables

| Variable | Required | Purpose | Example |
| --- | --- | --- | --- |
| `APP_VERSION` | No | Exposed FastAPI version string | `2.0.0` |
| `ALGORITHM_VERSION` | No | Saved in analysis sessions for traceability | `zone-measurement-2.0.0` |
| `CORS_ALLOW_ORIGINS` | Yes for hosted use | Comma-separated allowed frontend origins | `https://app.example.com` |
| `MAX_UPLOAD_BYTES` | No | Upload size limit in bytes | `12582912` |
| `STORAGE_ROOT` | Yes for hosted use | Persistent directory for uploaded images and analysis JSON | `/srv/zone-measurement/data/analysis_runs` |
| `OCR_MIN_CONFIDENCE` | No | Lower threshold for probable OCR matches | `0.72` |
| `OCR_HIGH_CONFIDENCE` | No | High-confidence OCR threshold | `0.88` |
| `OCR_SUGGESTION_CONFIDENCE` | No | Lower threshold for suggestion-only OCR | `0.52` |
| `OCR_MIN_MARGIN` | No | Score margin between top OCR candidates | `0.12` |
| `MEASUREMENT_MIN_CONFIDENCE` | No | Lower threshold for confident measurement acceptance | `0.6` |
| `MAX_DISCS` | No | Upper bound for detected discs in one image | `24` |
| `ANALYSIS_WORKERS` | No | Number of background analysis worker threads | `2` |
| `HF_API_KEY` | No | Present in config but not used in the active API path | empty |
| `OPENAI_API_KEY` | No | Present in config but not used in the active API path | empty |
| `USE_HF_VISION` | No | Present in config but not used in the active API path | `false` |
| `USE_OPENAI_VISION` | No | Present in config but not used in the active API path | `false` |

### Frontend Variable

| Variable | Required | Purpose | Example |
| --- | --- | --- | --- |
| `BACKEND_BASE_URL` | Yes for hosted use | Compile-time backend base URL for Flutter builds | `https://api.example.com` |

The frontend variable is passed with `--dart-define=BACKEND_BASE_URL=...`.

## Local Run Commands

### Backend
```powershell
cd C:\Users\najah\Desktop\Zone_Measurement_App\Backend
venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

### Frontend
```powershell
cd C:\Users\najah\Desktop\Zone_Measurement_App\frontend
flutter pub get
flutter run --dart-define=BACKEND_BASE_URL=http://127.0.0.1:8000
```

### Tests
```powershell
cd C:\Users\najah\Desktop\Zone_Measurement_App
Backend\venv\Scripts\python.exe -m unittest C:\Users\najah\Desktop\Zone_Measurement_App\Backend\tests\test_analysis_pipeline.py -v
Backend\venv\Scripts\python.exe -m compileall C:\Users\najah\Desktop\Zone_Measurement_App\Backend\app
cd frontend
flutter test test/analysis_session_model_test.dart
```

## Backend Hosting

### Install And Prepare
On a Linux host, provision:
- Python `3.12.x`
- Tesseract OCR
- Nginx if you plan to reverse proxy

Example Ubuntu package installation:
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip tesseract-ocr nginx
```

### Deploy The Backend Code
```bash
cd /srv
sudo mkdir -p zone-measurement
sudo chown $USER:$USER zone-measurement
git clone <your-repo-url> /srv/zone-measurement
cd /srv/zone-measurement/Backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
mkdir -p /srv/zone-measurement/data/analysis_runs
```

### Backend Environment File
Example `/etc/zone-measurement/backend.env`:
```env
APP_VERSION=2.0.0
ALGORITHM_VERSION=zone-measurement-2.0.0
CORS_ALLOW_ORIGINS=https://app.example.com
MAX_UPLOAD_BYTES=12582912
STORAGE_ROOT=/srv/zone-measurement/data/analysis_runs
OCR_MIN_CONFIDENCE=0.72
OCR_HIGH_CONFIDENCE=0.88
OCR_SUGGESTION_CONFIDENCE=0.52
OCR_MIN_MARGIN=0.12
MEASUREMENT_MIN_CONFIDENCE=0.6
MAX_DISCS=24
```

### Production Start Command
```bash
cd /srv/zone-measurement/Backend
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Suggested `systemd` Service
```ini
[Unit]
Description=Zone Measurement Backend
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/srv/zone-measurement/Backend
EnvironmentFile=/etc/zone-measurement/backend.env
ExecStart=/srv/zone-measurement/Backend/venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Backend Health Verification
- `GET https://api.example.com/`
- `GET https://api.example.com/docs`
- `GET https://api.example.com/api/openapi.json`

## Frontend Hosting

### Build The Flutter Web App
```bash
cd /srv/zone-measurement/frontend
flutter pub get
flutter build web --release --dart-define=BACKEND_BASE_URL=https://api.example.com
```

Build output:
- `frontend/build/web`

### Serve The Frontend
The Flutter web output can be served by:
- Nginx on the same VM
- a static host such as Netlify or Vercel
- an object storage static website host if you manage CORS and HTTPS separately

### Recommended Same-VM Nginx Layout
- `app.example.com` serves `frontend/build/web`
- `api.example.com` proxies to `127.0.0.1:8000`

Example backend proxy server block:
```nginx
server {
    server_name api.example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Example frontend static server block:
```nginx
server {
    server_name app.example.com;
    root /srv/zone-measurement/frontend/build/web;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

### Separate Hosting Option
If the frontend is hosted separately:
- build with `--dart-define=BACKEND_BASE_URL=https://api.example.com`
- set backend `CORS_ALLOW_ORIGINS=https://app.example.com`
- keep `STORAGE_ROOT` on the backend host

## Client UAT Deployment Steps
1. Deploy the backend on a persistent host with Tesseract installed.
2. Build and host the Flutter web frontend.
3. Set:
   - `BACKEND_BASE_URL` to the backend URL
   - `CORS_ALLOW_ORIGINS` to the frontend origin
4. Verify:
   - backend root responds
   - Swagger loads
   - frontend loads
5. Run a real image through the app.
6. Confirm that a new folder appears under `STORAGE_ROOT/<analysis_id>`.
7. Confirm `status.json` transitions from `queued` to `processing` to `completed`.
8. Save a review and confirm `analysis.json` and `result.json` update.
9. Export CSV through the backend export endpoint.
10. Restart the backend once during UAT and confirm any in-flight job moves to `failed` with an interrupted-job message rather than remaining stuck forever.

## Production Delivery Notes
- Use HTTPS for both app and API.
- Back up `STORAGE_ROOT` regularly. That directory is the operational record of uploads and review decisions.
- Rotate any secrets currently stored in local `.env` files before client delivery.
- Keep backend and frontend versioned together. The response payload contains fields that the current Flutter client expects.
- Roll back by redeploying the prior backend/frontend release pair and preserving the same storage volume.

## Post-Deployment Validation Checklist
- Frontend home screen loads over HTTPS
- Backend `/docs` loads
- Upload accepts a real plate image and `POST /api/analyze` returns quickly with an `analysis_id`
- `GET /api/analyze/{analysis_id}/status` reaches `completed`
- If the backend is restarted mid-job, the same status endpoint returns `failed` with a retry message instead of hanging in `processing`
- `GET /api/analyze/{analysis_id}/result` returns the final saved session
- Save review updates `final_code` and `final_diameter_mm`
- `GET /api/analysis/{analysis_id}/export` returns CSV
- A sample real plate still shows detected discs and saved JSON under `STORAGE_ROOT`

## Deployment Risks Still Outside The Current Repo
- Android release configuration still needs a real application ID and release signing before mobile store distribution.
- iOS release packaging still needs camera and photo-library usage descriptions in `Info.plist`.
- The repo has no Dockerfile, compose stack, or CI pipeline today.
- The repo currently stores a local `.env`; move real secrets into your host or secret manager before deployment.
