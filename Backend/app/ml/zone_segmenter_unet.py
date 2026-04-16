try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not installed. ML zone segmentation disabled.")

from PIL import Image
import numpy as np
import os
import cv2
from typing import Optional, Tuple


class ZoneSegmenterUNet:
    """
    U-Net based inhibition zone segmenter.
    Falls back to None if model is missing.
    """
    def __init__(self, model_path: str = "app/ml/models/zone_model.pt"):
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else "cpu"
        self._load_model()

    def _load_model(self):
        if not TORCH_AVAILABLE:
             return
             
        if not os.path.exists(self.model_path):
            print(f"⚠️ ML Model not found at {self.model_path}. ML segmentation disabled.")
            return

        try:
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
            print(f"✅ Loaded Zone Segmentation Model from {self.model_path} on {self.device}")
        except Exception as e:
            print(f"❌ Failed to load Zone Segmentation Model: {e}")
            self.model = None

    def segment(self, disc_roi: np.ndarray) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """
        Segments the zone from a cropped ROI containing the disc.
        Returns: (Zone diameter in pixels, thresholded mask).
        Returns: (None, None) if model not loaded or error.
        """
        if self.model is None or disc_roi is None or disc_roi.size == 0:
            return None, None

        orig_h, orig_w = disc_roi.shape[:2]

        try:
            # Standard U-Net Preprocessing (Resize to fixed size, e.g., 128x128)
            input_size = (128, 128)
            pil_img = Image.fromarray(cv2.cvtColor(disc_roi, cv2.COLOR_BGR2RGB) if len(disc_roi.shape) == 3 else disc_roi)
            
            transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
            ])
            
            img_tensor = transform(pil_img).to(self.device).unsqueeze(0)

            with torch.no_grad():
                output = self.model(img_tensor)
                
                # Assuming output is logits, apply sigmoid
                probs = torch.sigmoid(output).squeeze().cpu().numpy()
                
            # Resize back to original ROI size
            probs_resized = cv2.resize(probs, (orig_w, orig_h))
            
            # Threshold to get binary mask (1 = Zone, 0 = Background/Bacteria)
            binary_mask = (probs_resized > 0.5).astype(np.uint8) * 255
            
            # Find connected components to isolate the main zone
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            
            if num_labels <= 1: # Only background found
                 return 0.0, binary_mask
                 
            # Find the component closest to the center of the ROI
            center_x, center_y = orig_w / 2.0, orig_h / 2.0
            best_label = 1
            min_dist = float('inf')
            
            for i in range(1, num_labels):
                cx, cy = centroids[i]
                dist = (cx - center_x)**2 + (cy - center_y)**2
                if dist < min_dist:
                    min_dist = dist
                    best_label = i
                    
            # Isolate the main zone mask
            main_zone_mask = (labels == best_label).astype(np.uint8) * 255
            
            # Compute equivalent diameter based on area
            area = stats[best_label, cv2.CC_STAT_AREA]
            diameter = 2 * np.sqrt(area / np.pi)
                
            return float(diameter), main_zone_mask

        except Exception as e:
            print(f"❌ ML Segmentation Failed: {e}")
            return None, None
