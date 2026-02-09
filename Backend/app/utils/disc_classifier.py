import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple

# KNOWN PANEL CONFIGURATION (Hardcoded for Phase 6.1 objective)
# In production, this would be passed from request or DB
KNOWN_PANEL = [
    {"code": "P", "strength": 5, "prior_zone": "SMALL"},   # Penicillin (Often < 20mm or 0)
    {"code": "SXT", "strength": 25, "prior_zone": "MEDIUM"}, # Trimethoprim-Sulphamethoxazole
    {"code": "FOX", "strength": 30, "prior_zone": "XLARGE"}, # Cefoxitin (Very Large, > 30mm)
    {"code": "TE", "strength": 30, "prior_zone": "LARGE"},   # Tetracycline
    {"code": "DO", "strength": 30, "prior_zone": "LARGE"},   # Doxycycline
    {"code": "LZD", "strength": 30, "prior_zone": "LARGE"},  # Linezolid
]

def calculate_cost(disc_feat: Dict, panel_item: Dict, zone_mm: float) -> float:
    """
    Calculates cost (dissimilarity) between a detected disc and a panel item.
    Lower cost = Better match.
    """
    cost = 100.0 # Base cost
    
    # 1. STRENGTH CHECK (Highest Priority)
    feat_strength = disc_feat.get("strength")
    panel_strength = panel_item["strength"]
    
    if feat_strength is not None:
        if feat_strength == panel_strength:
            cost -= 50.0 # Huge bonus for strength match
        else:
            cost += 1000.0 # Effective Veto: Strength mismatch is almost impossible
            
    # 2. TEXT CHECK (Medium Priority)
    # Check if partial text matches code characters
    detected_text = disc_feat.get("partial_text", "")
    target_code = panel_item["code"]
    
    match_count = 0
    for char in detected_text:
        if char in target_code:
            match_count += 1
            
    if match_count > 0:
        cost -= (match_count * 10.0) # Bonus for matching letters
        
    # 3. BIOLOGICAL / ZONE SIZE PRIORS (Tie-Breaker)
    # Only applies if zone measurement is valid/available
    if zone_mm is not None and zone_mm > 0:
        prior = panel_item["prior_zone"]
        
        if prior == "SMALL":
            # P 5 is usually small/resistant. 
            if zone_mm < 15.0: cost -= 20.0 # Good fit
            if zone_mm > 35.0: cost += 50.0 # Unlikely (P usually not huge)
            
        elif prior == "XLARGE":
            # FOX 30 usually distinctively large?
            if zone_mm > 25.0: cost -= 15.0
            if zone_mm < 10.0: cost += 20.0 # Unlikely to be tiny
            
        elif prior == "MEDIUM":
             if 15.0 <= zone_mm <= 30.0: cost -= 10.0
             
    return cost

def classify_discs_globally(discs_features: List[Dict], zones_mm: List[float]) -> List[Dict]:
    """
    Performs global assignment of KNOWN_PANEL to detected discs.
    Returns list of assigned codes matching the input order.
    """
    num_discs = len(discs_features)
    num_panel = len(KNOWN_PANEL)
    
    # Handle mismatches in count (e.g. detected 7 discs but panel has 6)
    # We create a cost matrix of size (max, max) and maximize matches.
    
    # Actually, linear_sum_assignment handles non-square matrices but we want to ensure
    # we don't crash.
    
    # Assume: Detected Discs <= Panel Size (Ideal)
    # If Detected > Panel, some discs will be unmatched (or duplicate assignment?)
    # Constraint: "Assume A plate uses a known disc panel ... Total disc count is small"
    
    # We will try to assign unique Panel Items to Discs.
    if num_discs == 0:
        return []
        
    # Cost Matrix: Rows=Discs, Cols=Panel Items
    cost_matrix = np.zeros((num_discs, num_panel))
    
    for r in range(num_discs):
        for c in range(num_panel):
            cost_matrix[r, c] = calculate_cost(
                discs_features[r], 
                KNOWN_PANEL[c],
                zones_mm[r] if r < len(zones_mm) else 0.0
            )
            
    # Solve Assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Construct Result
    results = [None] * num_discs
    
    for i in range(len(row_ind)):
        disc_idx = row_ind[i]
        panel_idx = col_ind[i]
        
        # Check cost validity
        # If cost is super high (e.g. strength mismatch), is it better to return UNKNOWN?
        # User wants "Correctly identify... even when OCR unreadable".
        # But if Strength=5 and we matched to Strength=30, that's bad.
        
        final_cost = cost_matrix[disc_idx, panel_idx]
        
        assigned_item = KNOWN_PANEL[panel_idx]
        
        # Confidence Estimation
        # Base confidence 1.0, minus cost factors (normalized)
        # 100 is base cost. Min cost could be 100 - 50 - 30 ~ 20.
        # Max cost ~ 1000+.
        # Simple heuristic:
        confidence = 0.95
        if final_cost > 500: 
            confidence = 0.5 # Mismatch
        elif final_cost > 100:
            confidence = 0.8
            
        results[disc_idx] = {
            "code": assigned_item["code"],
            "confidence": confidence,
            "source": "Global-Classification"
        }
        
    # Fill any unassigned (unlikely with linear_sum_assignment logic unless shape weird)
    for i in range(num_discs):
        if results[i] is None:
            results[i] = {"code": "UNKNOWN", "confidence": 0.0, "source": "None"}
            
    return results
