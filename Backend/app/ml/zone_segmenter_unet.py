import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import cv2
from typing import Optional

class ZoneSegmenterUNet:
    """
    U-Net based inhibition zone segmenter.
    Falls back to None if model is missing.
    """
    def __init__(self, model_path: str = "app/ml/models/zone_model.pt"):
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"⚠️ ML Model not found at {self.model_path}. ML segmentation disabled.")
            return

        try:
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
            print(f"✅ Loaded Zone Segmentation Model from {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to load Zone Segmentation Model: {e}")
            self.model = None

    def segment(self, image: np.ndarray, disc_roi: np.ndarray) -> Optional[float]:
        """
        Segments the zone from a cropped ROI containing the disc.
        Returns: Zone diameter in pixels.
        Returns: None if model not loaded.
        """
        if self.model is None:
            return None

        try:
            # Standard U-Net Preprocessing (Resize to fixed size, e.g., 128x128)
            input_size = (128, 128)
            pil_img = Image.fromarray(disc_roi)
            
            transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
            ])
            
            img_tensor = transform(pil_img).to(self.device).unsqueeze(0)

            with torch.no_grad():
                output_mask = self.model(img_tensor)
                # Post-process: Threshold -> Resize back to original -> Measure
                
                # output_mask = torch.sigmoid(output_mask) > 0.5
                # ... measurement logic ...
                
                # Returning None to enforce fallback in this skeleton
                return None

        except Exception as e:
            print(f"❌ ML Segmentation Failed: {e}")
            return None
