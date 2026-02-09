import cv2
import numpy as np

def load_image_from_bytes(data: bytes) -> np.ndarray:
    """
    Decodes an image from bytes (e.g., from an API upload) into an OpenCV NumPy array.
    """
    try:
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        return img
    except Exception as e:
        raise ValueError(f"Image loading failed: {str(e)}")
