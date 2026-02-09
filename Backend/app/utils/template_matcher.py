import cv2
import numpy as np
import os

TEMPLATE_DIR = "app/assets/templates" # Placeholder path

def match_template(roi: np.ndarray) -> dict:
    """
    Matches ROI against known antibiotic disc templates.
    Returns: {"code": str, "confidence": float}
    """
    if roi.size == 0:
         return {"code": "", "confidence": 0.0}
         
    # Preprocess ROI (Grayscale)
    if len(roi.shape) == 3:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi
        
    best_match_code = ""
    best_match_score = 0.0
    
    if not os.path.exists(TEMPLATE_DIR):
        # Fallback if no templates exist yet
        return {"code": "NONE", "confidence": 0.0}
        
    for filename in os.listdir(TEMPLATE_DIR):
        if not filename.endswith(".png"):
            continue
            
        # Filename format: CODE_SIZE.png (e.g. FOX_30.png)
        code = filename.split("_")[0]
        
        template_path = os.path.join(TEMPLATE_DIR, filename)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        
        if template is None:
            continue
            
        # Resize template to match ROI size (roughly)?
        # Or Multi-scale matching.
        # For simplicity in Phase 6, assuming templates are roughly same scale (px/mm calibration).
        
        # Resize temp to match roi shape if needed, or match inside.
        if template.shape[0] > roi_gray.shape[0] or template.shape[1] > roi_gray.shape[1]:
            # Template bigger than ROI -> Skip or resize
            continue
            
        # Match
        res = cv2.matchTemplate(roi_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val > best_match_score:
            best_match_score = max_val
            best_match_code = code
            
    return {"code": best_match_code, "confidence": float(best_match_score)}
