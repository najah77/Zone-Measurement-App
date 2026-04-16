import cv2
import numpy as np

def apply_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts image to grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Applies Gaussian Blur to reduce noise."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_clahe(image: np.ndarray, clip_limit: float = 3.0, tile_grid_size: int = 8) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Optimized for AST plates to enhance inhibition zone boundaries.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(image)

def apply_adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """Applies Adaptive Thresholding to create a binary mask."""
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

def preprocess_image(image: np.ndarray) -> dict:
    """
    Runs the full preprocessing pipeline.
    Returns a dictionary with various stages for debugging/different detection steps.
    """
    gray = apply_grayscale(image)
    blurred = apply_gaussian_blur(gray)
    enhanced = apply_clahe(blurred)
    # Binary mask can be useful for contour detection
    binary = apply_adaptive_threshold(enhanced)
    
    return {
        "original": image,
        "gray": gray,
        "blurred": blurred,
        "enhanced": enhanced,
        "binary": binary
    }

def preprocess_plate(image: np.ndarray) -> np.ndarray:
    """
    Robust preprocessing pipeline for real-world AST images.
    Enhances contrast, normalizes lighting, and preserves edges.
    """
    if image is None or image.size == 0:
        return image
        
    # 1. Convert to LAB color space
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
    else:
        l = image
        a = None
        b = None

    # 2. Apply CLAHE to L channel to handle uneven lighting
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # 3. Merge channels back
    if len(image.shape) == 3:
        merged = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    else:
        enhanced = cl

    # 4. Apply bilateral filter to preserve edges but smooth textures like bacterial lawns
    # d=9, sigmaColor=75, sigmaSpace=75 are standard robust values
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # 5. Mild Gaussian blur to reduce high-frequency noise
    filtered = cv2.GaussianBlur(filtered, (3, 3), 0)

    return filtered
