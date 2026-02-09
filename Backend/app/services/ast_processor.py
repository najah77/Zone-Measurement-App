from app.utils.image_loader import load_image_from_bytes
from app.ml.inference import hybrid_analysis_pipeline
from app.models.schemas import AnalysisResponse

async def process_ast_image(image_bytes: bytes) -> AnalysisResponse:
    """
    Orchestrates the AST analysis pipeline (Phase 5).
    Delegates logic to the Hybrid ML/CV inference engine.
    """
    # 1. Load Image
    image = load_image_from_bytes(image_bytes)
    
    # 2. Run Hybrid Pipeline
    # Encapsulates Detection (ML/CV) -> Calibration -> Zone Measurement (ML/CV)
    results = hybrid_analysis_pipeline(image)
        
    return results
