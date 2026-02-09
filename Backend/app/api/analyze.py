from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import AnalysisResponse
from app.services.ast_processor import process_ast_image

router = APIRouter()

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(image: UploadFile = File(...)):
    print("Received:", image.filename, image.content_type)

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    image_bytes = await image.read()
    return await process_ast_image(image_bytes)
