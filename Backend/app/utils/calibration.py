from typing import List, Tuple
import numpy as np

STANDARD_DISC_DIAMETER_MM = 6.0

def calculate_mm_per_pixel(discs: List[Tuple[float, float, float]]) -> float:
    """
    Calculates the millimeter-per-pixel ratio based on detected antibiotic discs.
    Assumes all discs have a standard diameter of 6mm.
    
    Args:
        discs: List of (x, y, radius_px) tuples.
        
    Returns:
        mm_per_pixel factor.
    """
    if not discs:
        return 1.0 # Default fallback
    
    # Extract radii
    radii = [r for _, _, r in discs]
    
    # Use median to be robust against outliers (e.g., false positive small circles)
    avg_radius_px = np.median(radii)
    
    if avg_radius_px <= 0:
        return 1.0
        
    avg_diameter_px = avg_radius_px * 2
    mm_per_pixel = STANDARD_DISC_DIAMETER_MM / avg_diameter_px
    
    return mm_per_pixel

def pixels_to_mm(pixels: float, mm_per_pixel: float) -> float:
    """Converts pixel value to millimeters."""
    return pixels * mm_per_pixel
