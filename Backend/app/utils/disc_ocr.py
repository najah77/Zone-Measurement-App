import cv2
import numpy as np
import pytesseract
import re

# Set Tesseract Path (Modify if needed for Windows environment)
# Typically: C:\Program Files\Tesseract-OCR\tesseract.exe
# We'll try to rely on PATH or basic installation. 
# If verification fails, we might need a config.

def preprocess_for_ocr(roi: np.ndarray) -> np.ndarray:
    """
    Prepares disc image for OCR: Grayscale -> Upscale -> Threshold.
    """
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
        
    # Upscale for better character recognition (2x)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # Contrast Enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Thresholding (Otsu) to separate text from white disc background
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological cleaning (optional)
    # kernel = np.ones((2,2), np.uint8)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return thresh

def read_disc_code(roi: np.ndarray) -> dict:
    """
    Reads code from disc ROI trying multiple rotations.
    Returns: {"raw_text": str, "confidence": float}
    """
    if roi.size == 0:
        return {"raw_text": "", "confidence": 0.0}
        
    processed_base = preprocess_for_ocr(roi)
    
    best_text = ""
    best_conf = 0.0
    
    # Try 4 rotations: 0, 90, 180, 270
    for angle in [0, 90, 180, 270]:
        if angle == 0:
            rotated = processed_base
        elif angle == 90:
            rotated = cv2.rotate(processed_base, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(processed_base, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(processed_base, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        # OCR Configuration
        # --psm 6: Assume a single uniform block of text.
        # whitelist: A-Z (caps) and 0-9
        config = r'--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        try:
            data = pytesseract.image_to_data(rotated, config=config, output_type=pytesseract.Output.DICT)
            
            # image_to_data returns lists of words. We need to aggregate valid ones.
            # We look for matches that look like Antibiotic codes (e.g. FOX, TE, 30)
            
            n_boxes = len(data['text'])
            current_str = ""
            current_conf_sum = 0.0
            current_words = 0
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                if conf > 0 and len(text) > 0:
                    current_str += text + " "
                    current_conf_sum += conf
                    current_words += 1
            
            if current_words > 0:
                avg_conf = (current_conf_sum / current_words) / 100.0 # Normalize 0-1
                full_text = current_str.strip()
                
                if avg_conf > best_conf:
                    best_conf = avg_conf
                    best_text = full_text
                    
        except pytesseract.TesseractNotFoundError:
             print("⚠️ Tesseract OCR binary not found. Please install Tesseract-OCR.")
             break # No need to try rotations if binary is missing
        except Exception as e:
            # Tesseract might not be installed or configured
            # print(f"OCR Error: {e}")
            continue
            
    return {"raw_text": best_text, "confidence": best_conf}
