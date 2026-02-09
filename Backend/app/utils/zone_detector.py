import cv2
import numpy as np
from typing import List, Tuple, Optional
from app.utils.preprocessing import apply_clahe, apply_grayscale, apply_gaussian_blur

def get_radial_profile(img: np.ndarray, cx: int, cy: int, angle_deg: float, max_r: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts intensity profile along a ray from (cx, cy) at angle_deg.
    Returns: (radii, intensities)
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Sample points along the ray using vectorized spacing for speed
    r_values = np.arange(0, max_r, 1) # 1px steps
    x_values = np.clip(cx + r_values * cos_a, 0, img.shape[1]-1).astype(int)
    y_values = np.clip(cy + r_values * sin_a, 0, img.shape[0]-1).astype(int)
    
    intensities = img[y_values, x_values]
    return r_values, intensities

def detect_edge_in_profile(intensities: np.ndarray, min_r: int) -> int:
    """
    Finds index of sharpest intensity change (gradient) after min_r.
    Returns the index in the 'intensities' array (which roughly corresponds to radius).
    """
    # Simple gradient (derivative)
    # Smooth first to reduce noise from bacterial colonies or grain
    kernel_size = 5
    if len(intensities) < kernel_size:
        return -1
    
    # Gaussian smooth the profile
    smoothed = cv2.GaussianBlur(intensities.astype(float).reshape(-1, 1), (kernel_size, kernel_size), 0).flatten()
    gradient = np.gradient(smoothed)
    
    # Look for max gradient magnitude
    # Zones can be clearer (darker) or turbid (lighter) depending on plate type/lighting.
    # The edge is the transition.
    
    # ROI: From min_r onwards
    if min_r >= len(gradient):
        return -1
        
    roi_gradient = np.abs(gradient[min_r:])
    if len(roi_gradient) == 0:
        return -1
        
    # Find peak in gradient
    peak_idx = np.argmax(roi_gradient) + min_r
    return peak_idx

def perform_local_clustering(roi: np.ndarray) -> np.ndarray:
    """
    Step 3: Local Pixel Classification (SWITCH-lite).
    Uses K-Means (k=2) to separate Inhibition from Bacteria.
    Returns: Binary Mask (1=Bacteria, 0=Inhibition/Disc).
    """
    # Flatten ROI for clustering
    pixel_values = roi.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)
    
    # Define criteria = ( type, max_iter = 10, epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # K-Means with k=2
    k = 2
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Reshape labels back to image
    labels_img = labels.reshape(roi.shape)
    
    # Classification Rule:
    # "Cluster with higher mean intensity variance -> bacteria" 
    # (Or usually, bacteria = granular/textured, zone = smooth)
    # We calculate variance of pixels belonging to each cluster.
    
    vars = []
    for i in range(k):
        mask = (labels_img == i)
        if np.sum(mask) == 0:
            vars.append(0)
            continue
        cluster_pixels = roi[mask]
        vars.append(np.var(cluster_pixels))
        
    bacteria_label = np.argmax(vars)
    
    # Create Binary Mask (1 for Bacteria)
    bacteria_mask = (labels_img == bacteria_label).astype(np.uint8)
    
    # Morphological opening to remove noise (optional but recommended for stability)
    kernel = np.ones((3,3), np.uint8)
    bacteria_mask = cv2.morphologyEx(bacteria_mask, cv2.MORPH_OPEN, kernel)
    
    return bacteria_mask

def compute_radial_occupancy(mask: np.ndarray, roi_cx: int, roi_cy: int, max_r: int) -> np.ndarray:
    """
    Step 4: Radial Bacterial Occupancy Profile.
    Computes I(r) = fraction of bacteria pixels at radius r.
    Returns: Array of fractions (0.0 to 1.0).
    """
    occupancy = []
    
    # Create coordinate grids
    y, x = np.indices(mask.shape)
    
    # Calculate distances from center
    dists = np.sqrt((x - roi_cx)**2 + (y - roi_cy)**2)
    
    # Bin by radius (integer)
    dists_int = dists.astype(int)
    
    for r in range(max_r):
        # Pixels at radius r (using a thin ring ~1px)
        # Using a slight boolean mask for robustness: r-0.5 <= d < r+0.5 roughly
        ring_mask = (dists_int == r)
        
        total_pixels = np.sum(ring_mask)
        
        if total_pixels == 0:
            occupancy.append(0.0) # No pixels at this radius (outside image?)
        else:
            bacteria_pixels = np.sum(mask[ring_mask])
            fraction = bacteria_pixels / total_pixels
            occupancy.append(fraction)
            
    return np.array(occupancy)

def detect_change_point(occupancy_profile: np.ndarray, min_r: int) -> Tuple[float, float]:
    """
    Step 5: Dual-Criterion Change-Point Detection.
    1. Sharp Edge: Rapid rise in bacterial occupancy.
    2. Plateau: Sustained high occupancy (fuzzy edge).
    Returns: (boundary_radius, confidence)
    """
    roi_len = len(occupancy_profile)
    
    # Smooth profile for stability
    smoothed = cv2.GaussianBlur(occupancy_profile.reshape(-1, 1).astype(np.float32), (5, 5), 0).flatten()
    
    # PASS 1: Sharp Edge (Standard SWITCH)
    # Looking for transition from Inhibition (< 0.2) to Bacteria (> 0.4)
    for r in range(min_r, roi_len - 5):
        val = smoothed[r]
        
        # Criterion 1: Sharp Rise
        if val > 0.35: # Lowered slightly from 0.4 to catch onset
            # Check stability: Next 5 pixels should avg > 0.5
            future = smoothed[r:r+5]
            if np.mean(future) > 0.5:
                # Found sharp edge
                # Check previous history (should be low)
                # prev = smoothed[max(min_r, r-5):r]
                # if np.mean(prev) < 0.3: ... assumption of clean zone
                
                return float(r), 0.90
                
    # PASS 2: Plateau Detection (Fuzzy Edge)
    # If no sharp edge, look for where "Bacteria Lawn" definitively starts.
    # We allow a transition region. We define the boundary as the point where
    # occupancy reaches a "High Uncertainty" level (e.g. 0.5) and stays there.
    for r in range(min_r, roi_len - 10):
        # Look for sustained block of > 0.6 (Majority Bacteria)
        future_block = smoothed[r:r+10]
        if np.mean(future_block) > 0.6:
            # This is definitely bacteria.
            # Backtrack finding the "start" of valid bacteria
            # We return 'r' as the conservative estimate of the zone edge.
            return float(r), 0.75 # Lower confidence for fuzzy edge
            
    # PASS 3: Sanity Check - Is the whole ROI Inhibition? (Huge Zone)
    # If mean occupancy at end of ROI is LOW (< 0.3), the zone extends beyond ROI.
    # We should return ROI limit (censored measurement).
    if np.mean(smoothed[-10:]) < 0.3:
        # Zone is larger than ROI
        return float(roi_len), 0.60
        
    return 0.0, 0.0

def detect_zones(image: np.ndarray, discs: List[Tuple[float, float, float]]) -> Tuple[List[float], List[float]]:
    """
    Step 6: Main Pipeline (SWITCH-lite).
    Returns: (diameters, confidences)
    """
    diameters = []
    confidences = []
    
    img_h, img_w = image.shape[:2]
    gray = apply_grayscale(image)
    
    for x, y, r in discs:
        cx, cy, disc_r = int(x), int(y), int(r)
        
        # Step 1: ROI Isolation (Extended 8.0x for large halos)
        roi_r = int(disc_r * 8.0) 
        
        # Crop bounds
        x1, y1 = max(0, cx - roi_r), max(0, cy - roi_r)
        x2, y2 = min(img_w, cx + roi_r), min(img_h, cy + roi_r)
        
        roi_gray = gray[y1:y2, x1:x2]
        
        # Valid ROI check
        if roi_gray.size == 0 or roi_gray.shape[0] < disc_r or roi_gray.shape[1] < disc_r:
            diameters.append(0.0)
            confidences.append(0.0)
            continue
            
        # Step 2: Preprocessing
        roi_enhanced = apply_clahe(roi_gray, clip_limit=3.0)
        roi_blurred = apply_gaussian_blur(roi_enhanced, 3) # Slightly stronger blur for noise
        
        # Step 3: Local Clustering
        bacteria_mask = perform_local_clustering(roi_blurred)
        
        # Step 4: Radial Occupancy
        roi_cx_rel = cx - x1
        roi_cy_rel = cy - y1
        max_dist = min(roi_cx_rel, roi_cy_rel, roi_gray.shape[1]-roi_cx_rel, roi_gray.shape[0]-roi_cy_rel)
        
        occupancy = compute_radial_occupancy(bacteria_mask, roi_cx_rel, roi_cy_rel, max_dist)
        
        # Step 5: Change-Point Detection
        # Search from Disc Edge + Margin
        min_search_r = int(disc_r * 1.1)
        
        boundary_r, conf = detect_change_point(occupancy, min_search_r)
        
        if boundary_r > 0:
            # Valid detection within ROI?
            diameters.append(boundary_r * 2.0)
            confidences.append(conf)
            
        else:
            # Fallback Check: Is the bacterial mask showing "No Zone"?
            # If occupancy is High immediately after min_search_r?
            immediate = occupancy[min_search_r:min_search_r+5]
            if np.mean(immediate) > 0.6:
                 # Resistant / No Zone
                 diameters.append(disc_r * 2.0)
                 confidences.append(0.90)
            else:
                 # True Failure
                 diameters.append(0.0)
                 confidences.append(0.0)
                 
    return diameters, confidences
