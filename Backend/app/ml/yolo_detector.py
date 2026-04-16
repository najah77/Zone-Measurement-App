import cv2
import numpy as np
import os
import torch
from typing import List, Tuple, Optional, Dict

# Attempt to load YOLO, fail gracefully if ultralytics is not installed
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics not installed. YOLO object detection disabled.")

class YOLODetector:
    """
    YOLOv8 based object detector for Petri dish and Antibiotic discs.
    Falls back to None if model weights are missing or ultralytics is not installed.
    """
    def __init__(self, model_path: str = "app/ml/models/yolo_ast_model.pt"):
        self.model_path = model_path
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        if not YOLO_AVAILABLE:
            return
            
        if not os.path.exists(self.model_path):
            print(f"⚠️ YOLO Model weights not found at {self.model_path}. Falling back to OpenCV.")
            return

        try:
            self.model = YOLO(self.model_path)
            # Send to CPU if no CUDA
            self.model.to(self.device)
            print(f"✅ Loaded YOLO Object Detection Model on {self.device}")
        except Exception as e:
            print(f"❌ Failed to load YOLO Model: {e}")
            self.model = None

    def detect(self, image: np.ndarray) -> Tuple[Optional[tuple], List[Tuple[float, float, float]]]:
        """
        Detects the petri dish and antibiotic discs.
        Returns:
            plate_bbox: (x1, y1, x2, y2) or None
            discs: List of (cx, cy, r) identical to the OpenCV format.
        """
        if self.model is None:
            return None, []

        try:
            # Predict
            results = self.model(image, verbose=False, device=self.device)
            
            plate_bbox = None
            discs = []
            
            if not results or len(results) == 0:
                 return None, []
                 
            for det in results[0].boxes:
                # cls 0 could be plate, 1 could be disc (assuming general mapping)
                # If we don't have explicit class mappings, we can guess by sizes
                # Or check det.cls
                class_id = int(det.cls[0].item())
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                conf = det.conf[0].item()
                
                # Assume class 0 = plate, class 1 = disc for a standard custom trained model
                # Or use heuristic: largest bounding box that is roughly square is the plate
                
                w = x2 - x1
                h = y2 - y1
                
                # Dynamic heuristic if classes are unknown
                if class_id == 0 or (w > image.shape[1] * 0.4 and h > image.shape[0] * 0.4):
                     # Update plate bbox if this one is bigger
                     if plate_bbox is None or (w*h) > ((plate_bbox[2]-plate_bbox[0]) * (plate_bbox[3]-plate_bbox[1])):
                          plate_bbox = (x1, y1, x2, y2)
                else: 
                     # It's a disc
                     cx = x1 + w / 2.0
                     cy = y1 + h / 2.0
                     r = (w + h) / 4.0
                     discs.append((cx, cy, r))

            return plate_bbox, discs

        except Exception as e:
            print(f"❌ YOLO Inference Failed: {e}")
            return None, []

# Singleton instance
yolo_detector = YOLODetector()

def detect_objects_yolo(image: np.ndarray) -> Tuple[Optional[tuple], List[Tuple[float, float, float]]]:
    return yolo_detector.detect(image)
