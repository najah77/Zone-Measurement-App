from typing import Optional

# Simplified CLSI standard dictionary for mm breakpoints (S >= X, R <= Y)
# S: Susceptible, I: Intermediate, R: Resistant
# Note: Values are illustrative standard examples for common antibiotics
CLSI_BREAKPOINTS = {
    "CIP": {"S": 21, "R": 15}, # Ciprofloxacin
    "FOX": {"S": 18, "R": 14}, # Cefoxitin
    "LZD": {"S": 21, "R": 20}, # Linezolid
    "TE":  {"S": 15, "R": 11}, # Tetracycline
    "DO":  {"S": 14, "R": 10}, # Doxycycline
    "VA":  {"S": 17, "R": 14}, # Vancomycin
    "NOR": {"S": 17, "R": 12}, # Norfloxacin
    "SXT": {"S": 16, "R": 10}, # Trimethoprim/Sulfamethoxazole
    "CAZ": {"S": 21, "R": 17}, # Ceftazidime
    "ATM": {"S": 21, "R": 15}, # Aztreonam
    "IPM": {"S": 23, "R": 19}, # Imipenem
    "AMP": {"S": 17, "R": 13}, # Ampicillin
    "P":   {"S": 29, "R": 28}, # Penicillin
}

def get_interpretation(code: str, diameter_mm: Optional[float]) -> str:
    """
    Returns 'S', 'I', or 'R' based on CLSI breakpoints for the given antibiotic code.
    If the code is unknown or mm is missing, returns 'UNKNOWN'.
    """
    if diameter_mm is None or diameter_mm <= 0:
        return "UNKNOWN"
        
    code_upper = code.upper()
    if code_upper not in CLSI_BREAKPOINTS:
        return "UNKNOWN"
        
    bps = CLSI_BREAKPOINTS[code_upper]
    
    if diameter_mm >= bps["S"]:
        return "S"
    elif diameter_mm <= bps["R"]:
        return "R"
    else:
        return "I"
