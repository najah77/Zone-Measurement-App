import cv2
import numpy as np
import re
from PIL import Image
from google import genai
from app.core.config import settings

# Initialize Gemini Client if key is available
client = None
if getattr(settings, "GEMINI_API_KEY", ""):
    client = genai.Client(api_key=settings.GEMINI_API_KEY)

MODEL_NAME = "gemini-2.0-flash"
PROMPT = """You are analyzing an antibiotic susceptibility test disc.

Look at the image and read the antibiotic code printed on the disc.

Return ONLY the short code.

Examples:
IPM
FOX
CAZ
SXT
SAM
TE
ATM"""

def read_disc_code_with_gemini(image: np.ndarray) -> dict:
    """
    Sends a cropped disc ROI image to Google Gemini Vision API.
    """
    if image is None or image.size == 0:
        return {"code": "UNKNOWN", "confidence": 0.0}

    # 1. Verify Gemini is configured
    if not client:
        print("⚠️ GEMINI_API_KEY not set or client failed. Gemini Vision disabled.")
        return {"code": "UNKNOWN", "confidence": 0.0}

    try:
        # 2. Resize and Format ROI
        roi = cv2.resize(image, (256, 256))
        if len(roi.shape) == 3 and roi.shape[2] == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        else:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            
        pil_img = Image.fromarray(roi)

        # 3. Generate Content
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[PROMPT, pil_img]
        )
        
        text = response.text if response and hasattr(response, 'text') else ""

        if not text:
             return {"code": "UNKNOWN", "confidence": 0.0}

        # 4. Clean text
        code = re.sub(r'[^A-Z]', '', text.upper())

        # 5. Validate code length
        if 2 <= len(code) <= 4:
            return {"code": code, "confidence": 0.95}
        else:
            return {"code": "UNKNOWN", "confidence": 0.0}

    except Exception as e:
        print(f"⚠️ Error reading from Gemini API: {e}")
        return {"code": "UNKNOWN", "confidence": 0.0}
