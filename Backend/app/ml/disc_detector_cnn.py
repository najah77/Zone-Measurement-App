import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from typing import List, Dict, Optional

class DiscDetectorCNN:
    """
    CNN-based antibiotic disc detector.
    Falls back to None if model is missing or fails.
    """
    def __init__(self, model_path: str = "app/ml/models/disc_model.pt"):
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"⚠️ ML Model not found at {self.model_path}. ML detection disabled.")
            return

        try:
            # Assumes a standard detection model (e.g., Faster R-CNN or SSD)
            # For this skeleton, we assume a TorchScript model or state dict.
            # Using TorchScript for safer loading without definition.
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
            print(f"✅ Loaded Disc Detection Model from {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to load Disc Detection Model: {e}")
            self.model = None

    def detect(self, image: np.ndarray, confidence_threshold: float = 0.85) -> Optional[List[Dict]]:
        """
        Runs inference on the image.
        Returns: List of dicts {'x': float, 'y': float, 'r': float, 'confidence': float}
        Returns: None if model is not loaded (triggering fallback).
        """
        if self.model is None:
            return None

        try:
            # Preprocessing
            # Convert OpenCV (BGR) -> PIL -> Tensor
            pil_img = Image.fromarray(image[:, :, ::-1]) # BGR to RGB
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            img_tensor = transform(pil_img).to(self.device).unsqueeze(0)

            # Inference
            with torch.no_grad():
                predictions = self.model(img_tensor)
                # Parse predictions (mock structure for generic detection model)
                # Expected format: [{'boxes': [...], 'scores': [...], 'labels': [...]}]
                
                # NOTE: This part depends heavily on the specific model architecture.
                # Here we outline the logic for a standard object detection output.
                
                result = []
                # Placeholder logic:
                # boxes = predictions[0]['boxes']
                # scores = predictions[0]['scores']
                
                # for box, score in zip(boxes, scores):
                #     if score > confidence_threshold:
                #         x1, y1, x2, y2 = box.tolist()
                #         cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                #         r = max(x2 - x1, y2 - y1) / 2
                #         result.append({'x': cx, 'y': cy, 'r': r, 'confidence': float(score)})
                
                # Since we don't have a real model, we return None to force fallback 
                # even if the file existed (unless we mock the output).
                return None 

        except Exception as e:
            print(f"❌ ML Inference Failed: {e}")
            return None
