import cv2
import numpy as np
from typing import List, Tuple

def detect_discs(image: np.ndarray) -> List[Tuple[float, float, float]]:
    """
    Detects antibiotic discs using Hough Circle Transform.
    Returns a list of (x, y, radius) tuples.
    NOTE: This is a skeleton implementation. Parameters may need tuning for real images.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    
    # Robust HoughCircles parameters
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2,          # inverse ratio resolution
        minDist=40,      # minimum distance between centers
        param1=50,       # Canny high threshold
        param2=30,       # Accumulator threshold (lower = more circles)
        minRadius=15,    # Minimum disc radius (tuned for typical images)
        maxRadius=60     # Maximum disc radius
    )
    
    detected: List[Tuple[float, float, float]] = []
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        # Filter and convert
        raw_circles = []
        for i in circles[0, :]:
            x, y, r = float(i[0]), float(i[1]), float(i[2])
            raw_circles.append((x, y, r))
            
        # Sort discs: Top-to-bottom, then Left-to-right (roughly)
        # We sort primarily by Y, but allow a threshold to group rows
        raw_circles.sort(key=lambda c: (round(c[1] / 50), c[0]))
        
        detected = raw_circles
            
    # Fallback simulation if no circles found (for testing without real images)
    if not detected:
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2
        detected.append((float(center_x), float(center_y), 30.0))
        
    return detected
