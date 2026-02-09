import sys
import os
import numpy as np
import cv2

# Add current directory to sys.path
sys.path.append(os.getcwd())

def create_synthetic_plate():
    # 500x500 image
    # Background (Agar/Zone): Darker, Smooth
    img = np.ones((500, 500), dtype=np.uint8) * 80 
    
    # Bacterial Lawn: Lighter, Textured (High Variance)
    # create full noise mask
    noise = np.random.normal(0, 15, (500, 500)).astype(np.uint8) # High noise
    lawn = np.ones((500, 500), dtype=np.uint8) * 160
    lawn = cv2.add(lawn, noise)
    
    # Create Zone mask (Circle in center)
    mask = np.zeros((500, 500), dtype=np.uint8)
    cv2.circle(mask, (250, 250), 60, 255, -1) # Zone Radius 60px (Diam 120px)
    
    # Combine: Where mask is 0 (Lawn), use Lawn. Where mask is 1 (Zone), use Background.
    # Inverse mask
    mask_inv = cv2.bitwise_not(mask)
    
    img_bkg = cv2.bitwise_and(img, img, mask=mask)
    img_lawn = cv2.bitwise_and(lawn, lawn, mask=mask_inv)
    
    final_img = cv2.add(img_bkg, img_lawn)
    
    # Add Disc (White, Smooth)
    cv2.circle(final_img, (250, 250), 20, 255, -1) # Radius 20px
    
    # Convert to BGR
    return cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)

def test_switch_lite():
    print("⏳ Testing SWITCH-lite Logic...")
    from app.utils.zone_detector import detect_zones
    
    img = create_synthetic_plate()
    # cv2.imwrite("debug_synthetic_switch.png", img)
    
    # Disc at 250, 250, radius 20
    # Expected Zone Diameter: 120px
    
    print("   Running detect_zones...")
    diameters, confidences = detect_zones(img, [(250.0, 250.0, 20.0)])
    
    print(f"   Results: Diameters={diameters}, Confidences={confidences}")
    
    if len(diameters) > 0:
        val = diameters[0]
        if 110 <= val <= 130:
            print(f"✅ PASS: Detected {val}px (Target 120px)")
            print(f"   Confidence: {confidences[0]}")
            if confidences[0] > 0.8:
                print("✅ High Confidence Detected")
            else:
                print("⚠️ Low Confidence")
        else:
            print(f"❌ FAIL: Detected {val}px (Target 120px)")
            
    # Test Failure Case (No Zone, Texture Everywhere)
    print("\n⏳ Testing Failure Case (No Zone)...")
    img_fail = np.ones((500, 500), dtype=np.uint8) * 160
    noise = np.random.normal(0, 15, (500, 500)).astype(np.uint8)
    img_fail = cv2.add(img_fail, noise)
    cv2.circle(img_fail, (250, 250), 20, 255, -1) # Disc only
    img_fail_bgr = cv2.cvtColor(img_fail, cv2.COLOR_GRAY2BGR)
    
    diams_fail, confs_fail = detect_zones(img_fail_bgr, [(250.0, 250.0, 20.0)])
    print(f"   Results: {diams_fail}, {confs_fail}")
    
    # Should likely fail to find a boundary or return low confidence
    if diams_fail[0] == 0.0 or confs_fail[0] < 0.5:
        print("✅ PASS: Correctly handled failure/no-zone")
    else:
        print(f"⚠️ WARNING: Detected something? {diams_fail[0]}")

if __name__ == "__main__":
    try:
        test_switch_lite()
    except Exception as e:
        print(f"❌ Error: {e}")
