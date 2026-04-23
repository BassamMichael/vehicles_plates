import os
from ultralytics import YOLO
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Union, Any
from .dataset_loader import DatasetLoader

class YOLOModel:
    def __init__(self, model_path: str = "model/best.pt", 
                 labels_path: str = "model/labels.txt",
                 dataset_yaml_path: str = "dataset/data.yaml"):
        self.model_path = model_path
        self.labels_path = labels_path
        self.dataset_yaml_path = dataset_yaml_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.class_names = []
        
        self.load_model()
        self.load_labels()

    def load_model(self):
        """
        Loads the YOLO model and sets it to the optimal available device.
        """
        if not os.path.exists(self.model_path):
            print(f"Warning: Model not found at {self.model_path}")
            return

        try:
            # ultralytics automatically handles device fallback, but we can be explicit
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def load_labels(self):
        """
        Loads class names dynamically.
        Priority:
        1. dataset/data.yaml
        2. model/labels.txt
        """
        # Priority 1: Check dataset/data.yaml
        loader = DatasetLoader(self.dataset_yaml_path)
        names = loader.get_class_names()
        
        if names and len(names) > 0:
            self.class_names = names
            print(f"Loaded {len(names)} labels from {self.dataset_yaml_path}")
            return

        # Priority 2: Check model/labels.txt
        if os.path.exists(self.labels_path):
            try:
                with open(self.labels_path, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines() if line.strip()]
                print(f"Loaded {len(self.class_names)} labels from {self.labels_path}")
                return
            except Exception as e:
                print(f"Error loading labels from {self.labels_path}: {e}")

        # Fallback to model's built-in names if available
        if self.model is not None and hasattr(self.model, 'names') and self.model.names:
            names_dict = self.model.names
            self.class_names = [names_dict[k] for k in sorted(names_dict.keys())]
            print("Loaded labels from the model's internal metadata.")
        else:
            print("Warning: No class labels found.")

    def predict(self, image: Union[Image.Image, np.ndarray], conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """
        Runs inference on the provided image and returns structured detections.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Cannot run inference.")

        # Run inference
        results = self.model(image, conf=conf_threshold, verbose=False)
        
        detections = []
        if not results:
            return detections
            
        result = results[0]
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            # Get box coordinates [x1, y1, x2, y2]
            coords = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            
            # Get class name
            if cls_id < len(self.class_names):
                cls_name = self.class_names[cls_id]
            elif hasattr(self.model, 'names') and cls_id in self.model.names:
                cls_name = self.model.names[cls_id]
            else:
                cls_name = f"Class_{cls_id}"

            detections.append({
                "box": coords,
                "confidence": conf,
                "class_id": cls_id,
                "class_name": cls_name
            })

        return detections
