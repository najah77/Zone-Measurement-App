import cv2
import numpy as np
import base64
import requests
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llava-llama3"
PROMPT = """You are analyzing an antibiotic susceptibility test (AST) plate.

The image contains a small antibiotic disc.

Read the text printed on the disc.

Return ONLY the antibiotic code.

Examples:
IPM
FOX
CAZ
SXT
SAM
TE
ATM

Return only the short code."""

def read_disc_code_with_ollama(image: np.ndarray) -> dict:
    """
    Sends a cropped disc ROI image to a local Ollama vision model.
    """
    if image is None or image.size == 0:
        return {"code": "UNKNOWN", "confidence": 0.0}

    # 1. Convert ROI image to PNG in memory
    success, encoded_image = cv2.imencode('.png', image)
    if not success:
        return {"code": "UNKNOWN", "confidence": 0.0}

    # 2. Base64 encode
    base64_image = base64.b64encode(encoded_image).decode('utf-8')

    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "images": [base64_image],
        "stream": False
    }

    try:
        # 3. Send POST request
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        
        # 4. Parse response
        data = response.json()
        text = data.get("response", "")

        # 5. Clean text
        code = re.sub(r'[^A-Z]', '', text.upper())

        # 6. Validate code length
        if 2 <= len(code) <= 4:
            return {"code": code, "confidence": 0.9}
        else:
            return {"code": "UNKNOWN", "confidence": 0.0}

    except requests.exceptions.Timeout:
        print("⚠️ Ollama request timed out.")
        return {"code": "UNKNOWN", "confidence": 0.0}
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Ollama request failed: {e}")
        return {"code": "UNKNOWN", "confidence": 0.0}
    except Exception as e:
        print(f"⚠️ Error reading from Ollama: {e}")
        return {"code": "UNKNOWN", "confidence": 0.0}
