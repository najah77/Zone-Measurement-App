from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    PROJECT_NAME: str = "AST Analyzer Backend"
    API_PREFIX: str = "/api"
    VERSION: str = os.getenv("APP_VERSION", "2.0.0")
    ALGORITHM_VERSION: str = os.getenv("ALGORITHM_VERSION", "zone-measurement-2.0.0")
    HF_API_KEY: str = os.getenv("HF_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    USE_HF_VISION: bool = _env_bool("USE_HF_VISION", False)
    USE_OPENAI_VISION: bool = _env_bool("USE_OPENAI_VISION", False)
    MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))
    STORAGE_ROOT: Path = Path(
        os.getenv("STORAGE_ROOT", str(Path(__file__).resolve().parents[2] / "data" / "analysis_runs"))
    )
    OCR_MIN_CONFIDENCE: float = float(os.getenv("OCR_MIN_CONFIDENCE", "0.72"))
    OCR_HIGH_CONFIDENCE: float = float(os.getenv("OCR_HIGH_CONFIDENCE", "0.88"))
    OCR_SUGGESTION_CONFIDENCE: float = float(os.getenv("OCR_SUGGESTION_CONFIDENCE", "0.52"))
    OCR_MIN_MARGIN: float = float(os.getenv("OCR_MIN_MARGIN", "0.12"))
    MEASUREMENT_MIN_CONFIDENCE: float = float(os.getenv("MEASUREMENT_MIN_CONFIDENCE", "0.6"))
    DISC_DIAMETER_MM: float = 6.0
    MAX_DISCS: int = int(os.getenv("MAX_DISCS", "24"))


settings = Settings()
