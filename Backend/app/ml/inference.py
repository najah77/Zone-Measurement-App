from typing import List, Dict, Tuple, Optional
from app.utils.disc_roi import extract_disc_roi
from app.utils.disc_ocr import read_disc_code
from app.utils.template_matcher import match_template
from app.utils.code_resolver import resolve_code
import numpy as np
import cv2
from app.ml.disc_detector_cnn import DiscDetectorCNN
from app.ml.zone_segmenter_unet import ZoneSegmenterUNet
from app.utils.disc_detector import detect_discs as detect_discs_cv
from app.utils.zone_detector import detect_zones as detect_zones_cv
from app.utils.disc_roi import extract_disc_roi
from app.utils.disc_ocr import read_disc_code
from app.utils.template_matcher import match_template
from app.utils.code_resolver import resolve_code
from app.utils.disc_features import extract_disc_features
from app.utils.disc_classifier import classify_discs_globally
from app.utils.calibration import calculate_mm_per_pixel, pixels_to_mm
from app.utils.measurement import map_disc_to_antibiotic, calculate_inhibition_result
from app.models.schemas import AnalysisResponse, AntibioticResult

# Initialize models once (singleton pattern)
# These will print warnings if models are missing, which is expected behavior.
disc_model = DiscDetectorCNN()
zone_model = ZoneSegmenterUNet()

def hybrid_analysis_pipeline(image: np.ndarray) -> AnalysisResponse:
    """
    Orchestrates the hybrid ML + OpenCV analysis pipeline (Phase 5.1).
    Prioritizes ML models; falls back to Phase 5.1 Radial Profiling logic.
    """
    
    # --- Step 1: Hybrid Disc Detection ---
    print("🔍 Step 1: Detecting Discs...")
    
    # Try ML Detection first
    ml_discs = disc_model.detect(image)
    
    final_discs = [] # List of (x, y, r) tuples
    detection_source = "ML"
    
    if ml_discs and len(ml_discs) > 0:
        print(f"   ✅ Using ML Disc Detection ({len(ml_discs)} discs found)")
        # Normalize to tuple format: (x, y, r)
        final_discs = [(d['x'], d['y'], d['r']) for d in ml_discs]
    else:
        print("   ⚠️ ML Disc Detection unavailable. Falling back to OpenCV Robust Hough.")
        # Fallback to Phase 4 Robust HoughCircles
        final_discs = detect_discs_cv(image)
        detection_source = "CV"
        
    # --- Step 2: Auto-Calibration ---
    print("📏 Step 2: Calibrating...")
    # Calculates mm/px assuming discs are standard 6mm
    mm_per_pixel = calculate_mm_per_pixel(final_discs)
    
    # --- Step 3: Zone Measurement ---
    print("⭕ Step 3: Measuring Zones (SWITCH-lite)...")
    
    # Run batch zone detection (Phase 5.2 SWITCH-lite Logic)
    # Returns (diameters, confidences)
    batch_diameters_px, batch_confidences = detect_zones_cv(image, final_discs)
    
    results = []
    
    # --- Phase 6.1: Global Classification Preparation ---
    batch_features = []
    batch_mm = []
    batch_measurement_data = [] # Store diameter_mm, status, meas_conf to reuse
    
    # Pass 1: Measure and Extract Features
    for i, (x, y, r) in enumerate(final_discs):
        disc_diameter_px = r * 2.0
        zone_diameter_px = batch_diameters_px[i]
        algo_confidence = batch_confidences[i]
        
        # 1. Measurement
        diameter_mm = None
        status = "failed"
        meas_conf = 0.0
        val_mm_for_class = 0.0
        
        if zone_diameter_px > 0:
            val_mm = calculate_inhibition_result(
                zone_diameter_px=zone_diameter_px,
                disc_diameter_px=disc_diameter_px,
                mm_per_pixel=mm_per_pixel
            )
            if val_mm > 0:
                diameter_mm = val_mm
                val_mm_for_class = val_mm
                status = "valid"
                # Confidence Tuning
                conf_base = algo_confidence
                if detection_source == "ML":
                    conf_base = min(1.0, conf_base + 0.05)
                meas_conf = round(conf_base, 2)
            else:
                status = "failed"
        
        batch_mm.append(val_mm_for_class)
        batch_measurement_data.append({
            "diameter_mm": diameter_mm,
            "status": status,
            "meas_conf": meas_conf
        })
        
        # 2. Feature Extraction
        roi = extract_disc_roi(image, x, y, r)
        feats = extract_disc_features(roi)
        
        # (Optional) Keep raw OCR for candidates listing if needed
        # ocr_res = read_disc_code(roi)
        # feats["ocr_res"] = ocr_res 
        batch_features.append(feats)
        
    # Pass 2: Global Classification
    classified_results = classify_discs_globally(batch_features, batch_mm)
    
    # Pass 3: Final Response Construction
    results = []
    for i in range(len(final_discs)):
        cls_res = classified_results[i]
        meas_data = batch_measurement_data[i]
        
        final_code = cls_res["code"]
        code_conf = cls_res["confidence"]
        
        # Candidates? Just list the classified one + partial text from features
        candidates = [final_code]
        if batch_features[i]["partial_text"]:
             candidates.append(f"OCR:{batch_features[i]['partial_text']}")
        
        # Needs Confirmation?
        needs_conf = True
        if code_conf > 0.9: 
            needs_conf = False
            
        results.append(AntibioticResult(
            code=final_code,
            code_confidence=code_conf,
            diameter_mm=meas_data["diameter_mm"],
            measurement_confidence=meas_data["meas_conf"],
            candidates=candidates,
            needs_confirmation=needs_conf,
            measurement_status=meas_data["status"]
        ))
        
    return AnalysisResponse(results=results)
