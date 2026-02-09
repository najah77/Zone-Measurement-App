import sys
import os
import numpy as np
import cv2

# Add current directory to sys.path
sys.path.append(os.getcwd())

try:
    print("⏳ Importing Zone Detection Logic (No ML)...")
    from app.utils.zone_detector import detect_zones
    from app.utils.preprocessing import apply_clahe
    
    print("⏳ Creating Synthetic AST Plate...")
    # Create a 500x500 image with a gradient-like zone to test edge detection
    img = np.ones((500, 500, 3), dtype=np.uint8) * 120 # Background (Agar) - Mid Grey
    
    # Draw a "Zone" (Darker than background)
    # 60px radius = 120px diameter
    cv2.circle(img, (250, 250), 60, (60, 60, 60), -1) 
    
    # Draw a "Disc" (White)
    # 20px radius = 40px diameter
    cv2.circle(img, (250, 250), 20, (255, 255, 255), -1) 
    
    # Add some noise
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    print("⏳ Running Detect Zones...")
    # Disc at 250,250 with radius 20
    # Expected result: ~120px diameter
    zones = detect_zones(img, [(250.0, 250.0, 20.0)])
    
    print(f"✅ Detect Zones Result: {zones}")
    
    if len(zones) > 0:
        val = zones[0]
        # Allow some margin errors due to synthetic drawing/noise
        if 110 <= val <= 130:
            print(f"✅ PASS: Result {val} is close to expected 120.0")
        else:
            print(f"❌ FAIL: Result {val} is far from expected 120.0")
            
except ImportError as e:
    print(f"❌ Import Failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Verification Failed: {e}")
    sys.exit(1)
