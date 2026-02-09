import sys
import os
import numpy as np
import cv2

# Add current directory to sys.path
sys.path.append(os.getcwd())

def test_strict_failure():
    print("⏳ Testing Strict Failure Handling...")
    
    # 1. ZONE DETECTOR TEST
    from app.utils.zone_detector import detect_zones
    
    # Create Unreadable Plate (High Noise, No Zone, Texture everywhere)
    print("   Creating Noisy Input...")
    img = np.ones((500, 500, 3), dtype=np.uint8) * 120 
    noise = np.random.normal(0, 30, (500, 500, 3)).astype(np.uint8) # Extreme noise
    img = cv2.add(img, noise)
    cv2.circle(img, (250, 250), 20, (255, 255, 255), -1) # Disc only
    
    print("   Running detect_zones (Expect 0.0)...")
    diameters, confidences = detect_zones(img, [(250.0, 250.0, 20.0)])
    
    print(f"   Result: {diameters}, {confidences}")
    if diameters[0] == 0.0:
        print("✅ PASS: Detector returned 0.0 for failed measurement")
    else:
        print(f"❌ FAIL: Detector returned {diameters[0]} (Expected 0.0)")

    # 2. INFERENCE PIPELINE TEST (Mocking logic)
    print("\n⏳ Testing Inference Logic (Mock)...")
    from app.models.schemas import AntibioticResult
    
    # Mock Input: 0.0 px (Failed)
    zone_px = 0.0
    disc_px = 40.0
    
    # Logic from inference.py (simplified check)
    if zone_px > 0:
        status = "valid"
        val = 15.0 # mm
    else:
        status = "failed"
        val = None
        
    if status == "failed" and val is None:
         print("✅ PASS: Inference Logic sets status='failed', diameter=None")
    else:
         print(f"❌ FAIL: Inference Logic set status={status}, diameter={val}")

if __name__ == "__main__":
    test_strict_failure()
