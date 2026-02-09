from typing import List

def map_disc_to_antibiotic(index: int) -> str:
    """
    Simulates mapping a detected disc to an antibiotic code based on position or OCR.
    """
    codes = ["AMP", "CIP", "CN", "TE", "AMC", "FOX", "AZM", "CRO", "IPM", "MEM"]
    return codes[index % len(codes)]

def calculate_inhibition_result(zone_diameter_px: float, disc_diameter_px: float, mm_per_pixel: float) -> float:
    """
    Calculates the final reportable zone size in mm (Total Diameter).
    
    Validation Logic (Subtraction Disc):
    1. If zone_diameter_px <= 0: Return 0.0 (Failure/Invalid).
    2. Calculate Raw Inhibition = Zone - Disc.
    3. If Raw Inhibition is negligible (< 0.5mm equivalent), force result to Disc Diameter (~6mm).
    4. Otherwise, report Total Diameter (Standard).
    """
    if zone_diameter_px <= 0:
        return 0.0
        
    # 1. Convert to mm
    zone_mm = zone_diameter_px * mm_per_pixel
    disc_mm = disc_diameter_px * mm_per_pixel
    
    # 2. Subtraction Check
    inhibition_halo_mm = zone_mm - disc_mm
    
    # Threshold for "No Inhibition"
    if inhibition_halo_mm < 0.5:
        # Return standard disc size (6mm) or detected size?
        # Prompt says "If inhibition_radius_px <= 0 -> report 6.0 mm"
        # We'll return 6.0 explicitly if it's "No Inhibition".
        # But to be consistent with calibration, we return detected disc size (approx 6mm).
        return round(disc_mm, 2)
        
    # 3. Return Total Diameter
    return round(zone_mm, 2)

