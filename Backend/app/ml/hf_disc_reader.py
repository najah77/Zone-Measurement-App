import cv2
import numpy as np
import requests
import re
from app.core.config import settings

HF_API_URL = "https://router.huggingface.co/hf-inference/models/Salesforce/blip-image-captioning-large"

PROMPT = """
Look at the image of an antibiotic susceptibility test disc.
Return ONLY the antibiotic code printed on the disc.
Examples: IPM FOX CAZ SXT SAM TE ATM
"""

def read_disc_code_with_hf(image: np.ndarray) -> dict:

    if image is None or image.size == 0:
        return {"code": "UNKNOWN", "confidence": 0.0}

    hf_key = getattr(settings, "HF_API_KEY", "")

    if not hf_key:
        print("⚠️ HF_API_KEY not configured.")
        return {"code": "UNKNOWN", "confidence": 0.0}

    try:

        roi = cv2.resize(image, (256, 256))

        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        success, encoded_image = cv2.imencode(".png", roi)

        if not success:
            return {"code": "UNKNOWN", "confidence": 0.0}

        headers = {
            "Authorization": f"Bearer {hf_key}",
            "Content-Type": "application/octet-stream"
        }

        response = requests.post(
            HF_API_URL,
            headers=headers,
            data=encoded_image.tobytes(),
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        text = ""

        if isinstance(data, list) and len(data) > 0:
            text = data[0].get("generated_text", "")

        code = re.sub(r"[^A-Z]", "", text.upper())

        if 2 <= len(code) <= 4:
            return {"code": code, "confidence": 0.9}

        return {"code": "UNKNOWN", "confidence": 0.0}

    except Exception as e:
        print(f"⚠️ Error reading from HuggingFace API: {e}")
        return {"code": "UNKNOWN", "confidence": 0.0}