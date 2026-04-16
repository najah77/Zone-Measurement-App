import cv2
import numpy as np
import base64
import re
from openai import OpenAI
from app.core.config import settings

PROMPT = """
Look at the image of an antibiotic susceptibility test disc.
Return ONLY the short antibiotic code printed on the disc.
Examples: IPM, FOX, CAZ, SXT, SAM, TE, ATM.
Return the letters only, no other text.
"""

def read_disc_code_with_openai(image: np.ndarray) -> dict:
    if image is None or image.size == 0:
        return {"code": "UNKNOWN", "confidence": 0.0}

    openai_key = getattr(settings, "OPENAI_API_KEY", "")
    if not openai_key:
        print("⚠️ OPENAI_API_KEY not configured.")
        return {"code": "UNKNOWN", "confidence": 0.0}

    try:
        client = OpenAI(api_key=openai_key)
        
        # Resize to save bandwidth and improve API latency
        roi = cv2.resize(image, (256, 256))
        
        # Convert BGR to RGB if needed
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
        success, buffer = cv2.imencode('.png', roi)
        if not success:
            return {"code": "UNKNOWN", "confidence": 0.0}
            
        base64_image = base64.b64encode(buffer).decode('utf-8')

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=10,
            temperature=0.0
        )
        
        text = response.choices[0].message.content.strip()
        code = re.sub(r"[^A-Z]", "", text.upper())

        if 2 <= len(code) <= 4:
            return {"code": code, "confidence": 0.9}

        return {"code": "UNKNOWN", "confidence": 0.0}

    except Exception as e:
        print(f"⚠️ Error reading from OpenAI API: {e}")
        return {"code": "UNKNOWN", "confidence": 0.0}
