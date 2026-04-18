from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api import analyze
from app.services.analysis_jobs import recover_interrupted_analysis_jobs

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url="/api/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS (Allow all for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.CORS_ALLOW_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include Routers
app.include_router(analyze.router, prefix=settings.API_PREFIX, tags=["Analysis"])


@app.on_event("startup")
def recover_analysis_jobs_on_startup() -> None:
    recover_interrupted_analysis_jobs()

@app.get("/")
def root():
    return {"message": "AST Analyzer Backend Running"}
