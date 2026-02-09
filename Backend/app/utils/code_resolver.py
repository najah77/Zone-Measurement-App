from typing import List, Dict, Optional
import difflib

# Known Antibiotics Dictionary (Expanded)
# Code -> Full Name
KNOWN_ANTIBIOTICS = {
    "FOX": "Cefoxitin",
    "TE": "Tetracycline",
    "SXT": "Trimethoprim-Sulfamethoxazole",
    "LZD": "Linezolid",
    "DO": "Doxycycline",
    "P": "Penicillin",
    "CN": "Gentamicin",
    "AMP": "Ampicillin",
    "CIP": "Ciprofloxacin",
    "AMC": "Amoxicillin-Clavulanate",
    "CRO": "Ceftriaxone",
    "IPM": "Imipenem",
    "MEM": "Meropenem"
}

def resolve_code(ocr_result: dict, template_result: dict) -> dict:
    """
    Resolves final code from OCR and Template Matching.
    Returns: {
        "final_code": str, 
        "confidence": float, 
        "candidates": List[str], 
        "needs_confirmation": bool,
        "source": str
    }
    """
    ocr_text = ocr_result.get("raw_text", "").upper().strip()
    ocr_conf = ocr_result.get("confidence", 0.0)
    
    tpl_code = template_result.get("code", "")
    tpl_conf = template_result.get("confidence", 0.0)
    
    # 1. Cleaning OCR Text to likely code
    # Extract only alphabetical parts for code matching?
    # Or matches like "FOX 30" -> "FOX"
    ocr_code_cand = ""
    for token in ocr_text.split():
        # Clean token (remove numbers)
        alpha = ''.join(filter(str.isalpha, token))
        if len(alpha) >= 1 and alpha in KNOWN_ANTIBIOTICS:
             ocr_code_cand = alpha
             break
    
    # Fuzzy matching against dictionary if exact match fails
    if not ocr_code_cand and len(ocr_text) > 1:
        # Try to find closest known code in OCR text
        matches = difflib.get_close_matches(ocr_text, KNOWN_ANTIBIOTICS.keys(), n=1, cutoff=0.5)
        if matches:
            ocr_code_cand = matches[0]
            
    # 2. Decision Logic
    final_code = "UNKNOWN"
    final_conf = 0.0
    source = "None"
    
    # Case A: Strong Agreement
    if ocr_code_cand and tpl_code and ocr_code_cand == tpl_code:
        final_code = ocr_code_cand
        final_conf = max(ocr_conf, tpl_conf)
        source = "Agreement"
        
    # Case B: Strong Template, Weak/No OCR
    elif tpl_conf > 0.8 and tpl_conf > ocr_conf:
        final_code = tpl_code
        final_conf = tpl_conf
        source = "Template"
        
    # Case C: Strong OCR, Weak/No Template
    elif ocr_conf > 0.7 and ocr_code_cand:
        final_code = ocr_code_cand
        final_conf = ocr_conf
        source = "OCR"
        
    # Case D: Ambiguous - List Candidates
    candidates = []
    if ocr_code_cand: candidates.append(ocr_code_cand)
    if tpl_code and tpl_code not in candidates: candidates.append(tpl_code)
    
    # Confirmation Check
    needs_conf = True
    if final_conf > 0.95 and final_code in KNOWN_ANTIBIOTICS:
        needs_conf = False
        
    return {
        "final_code": final_code,
        "confidence": round(final_conf, 2),
        "candidates": candidates,
        "needs_confirmation": needs_conf,
        "source": source
    }
