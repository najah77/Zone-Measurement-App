import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class DiscCNN(nn.Module):
    def __init__(self, num_classes: int):
        super(DiscCNN, self).__init__()
        # Input: 3 x 96 x 96 (RGB)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2) # Output: 32 x 48 x 48
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2) # Output: 64 x 24 x 24
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # No pool here based on architecture spec, but flatten requires size
        # Let's add an AdaptiveAvgPool or calculate flat size to be robust. 
        # Spec: Conv2D -> ReLU. Then Flatten. Output is 128 x 24 x 24
        
        # 128 * 24 * 24 = 73728
        self.fc1 = nn.Linear(128 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Note: CrossEntropyLoss applies Softmax internally during training.
        # But we will explicitly apply softmax during inference.
        return x

class DiscClassifier:
    """
    Production CNN classifier for AST antibiotic discs.
    Provides a wrapper for the PyTorch nn.Module.
    """
    def __init__(self, model_path: str = "app/ml/models/disc_classifier.pt", class_map_path: str = "app/ml/models/class_map.json"):
        self.model_path = model_path
        self.class_map_path = class_map_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.idx_to_class = {}
        
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Attempt to load model and class mappings
        self._load_resources()

    def _load_resources(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.class_map_path):
            print(f"⚠️ CNN Model resources not found at '{self.model_path}'. CNN classification disabled.")
            return

        try:
            with open(self.class_map_path, 'r') as f:
                class_to_idx = json.load(f)
                self.idx_to_class = {v: k for k, v in class_to_idx.items()}
                
            num_classes = len(self.idx_to_class)
            
            self.model = DiscCNN(num_classes=num_classes)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("✅ CNN Disc Classifier loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load CNN Disc Classifier: {e}")
            self.model = None

    def predict(self, roi_image: np.ndarray) -> dict:
        """
        Predicts antibiotic code from a 96x96 BGR/RGB numpy array ROI.
        Returns empty dict if model falls back.
        """
        if self.model is None or not self.idx_to_class:
            return {} # Signal inference.py to fallback to OCR
            
        if roi_image is None or roi_image.size == 0:
            return {"code": "UNKNOWN", "confidence": 0.0}
            
        # Convert OpenCV BGR to Pillow RGB
        if len(roi_image.shape) == 3 and roi_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
        elif len(roi_image.shape) == 2:
            rgb_image = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2RGB)
        else:
             rgb_image = roi_image
             
        pil_img = Image.fromarray(rgb_image)
        
        # Preprocess
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get max probability
            conf, predicted_idx = torch.max(probabilities, 1)
            
        idx = predicted_idx.item()
        confidence = conf.item()
        code = self.idx_to_class.get(idx, "UNKNOWN")
        
        return {
            "code": code,
            "confidence": round(confidence, 4)
        }
