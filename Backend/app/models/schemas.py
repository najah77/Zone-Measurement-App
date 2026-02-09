from pydantic import BaseModel
from typing import List, Optional

class AntibioticResult(BaseModel):
    """
    Represents a single antibiotic disc analysis result.
    """
    code: str
    code_confidence: float = 0.0
    diameter_mm: Optional[float]
    measurement_confidence: float
    candidates: List[str] = []
    needs_confirmation: bool = False
    measurement_status: str = "valid"


class AnalysisResponse(BaseModel):
    """
    Response schema for the analysis endpoint.
    """
    results: List[AntibioticResult]
