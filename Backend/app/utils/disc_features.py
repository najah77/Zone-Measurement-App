import cv2
import numpy as np
import pytesseract
import re
from typing import Optional, Dict

def extract_disc_features(roi: np.ndarray) -> Dict:
    """
    Extracts features from Disc ROI for classification.
    Key feature: Numeric Strength (5, 25, 30, etc.)
    Returns: {
        "strength": int or None,
        "partial_text": str (uppercase),
        "raw_ocr": str
    }
    """
    if roi.size == 0:
        return {"strength": None, "partial_text": "", "raw_ocr": ""}
        
    # Preprocessing tailored for numbers
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
        
    # Upscale
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    
    # Thresholding
    # Try Otsu
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # OCR Config - allow finding numbers specifically
    custom_config = r'--psm 6' 
    
    best_strength = None
    best_text = ""
    raw_ocr = ""
    
    # Rotations
    for angle in [0, 90, 180, 270]:
        if angle == 0:
            rotated = thresh
        elif angle == 90:
            rotated = cv2.rotate(thresh, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(thresh, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(thresh, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        try:
            text = pytesseract.image_to_string(rotated, config=custom_config)
            raw_ocr += text + " | "
            
            # Find Pattern: Number (5, 10, 25, 30)
            # Typically at end of string or separate line
            # Regex for isolated numbers or numbers mixed with codes
            
            # Look for specific known strengths
            nums = re.findall(r'\b(5|10|15|25|30)\b', text)
            if nums:
                # Found a valid strength!
                # Prefer the last one detected (often strength is below code)
                strength_val = int(nums[-1])
                best_strength = strength_val
            
            # Extract Text (Letters)
            # Filter for likely code letters
            letters = re.findall(r'[A-Z]', text)
            if letters:
                 current_text = "".join(letters)
                 if len(current_text) > len(best_text):
                     best_text = current_text
                     
        except:
             continue
             
    return {
        "strength": best_strength,
        "partial_text": best_text,
        "raw_ocr": raw_ocr
    }
