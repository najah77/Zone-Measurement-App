import cv2
import numpy as np

def extract_disc_roi(image: np.ndarray, x: float, y: float, r: float, expand_ratio: float = 0.15) -> np.ndarray:
    """
    Crops a circular ROI for the disc, masking out the background.
    
    Args:
        image: Source image (BGR or Gray).
        x, y: Center coordinates.
        r: Radius.
        expand_ratio: How much to expand the crop relative to radius (default 15%).
        
    Returns:
        np.ndarray: Cropped, masked image of the disc.
    """
    img_h, img_w = image.shape[:2]
    
    # Calculate expanded radius
    expanded_r = int(r * (1.0 + expand_ratio))
    
    # Crop bounds
    x1 = max(0, int(x - expanded_r))
    y1 = max(0, int(y - expanded_r))
    x2 = min(img_w, int(x + expanded_r))
    y2 = min(img_h, int(y + expanded_r))
    
    # Crop
    crop = image[y1:y2, x1:x2].copy()
    
    if crop.size == 0:
        return np.array([])
        
    # Masking
    # Create a mask of the same size as crop
    h, w = crop.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Center of the crop
    cx, cy = w // 2, h // 2
    
    # Draw white circle on mask (use the original radius, or slightly expanded?)
    # Valid text is ON the disc. So we should mask using the disc radius (r).
    # Since we expanded the crop, the disc is centered in the crop.
    
    # We use 'r' for masking to remove background noise (agar/zone).
    # Re-calculate 'r' in terms of pixels (it passed in as float)
    mask_r = int(r)
    
    cv2.circle(mask, (cx, cy), mask_r, 255, -1)
    
    # Apply mask
    if len(crop.shape) == 3:
        masked_crop = cv2.bitwise_and(crop, crop, mask=mask)
    else:
        masked_crop = cv2.bitwise_and(crop, crop, mask=mask)
        
    return masked_crop
