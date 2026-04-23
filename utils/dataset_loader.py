import os
import yaml
from typing import List, Optional

class DatasetLoader:
    def __init__(self, data_yaml_path: str):
        """
        Initializes the DatasetLoader with the path to the dataset's data.yaml.
        """
        self.data_yaml_path = data_yaml_path
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """
        Loads the YAML configuration from the dataset.
        """
        if not os.path.exists(self.data_yaml_path):
            return {}
        try:
            with open(self.data_yaml_path, 'r') as f:
                config = yaml.safe_load(f)
                return config if config else {}
        except Exception as e:
            print(f"Error loading dataset config from {self.data_yaml_path}: {e}")
            return {}

    def get_class_names(self) -> Optional[List[str]]:
        """
        Extracts class names from the dataset configuration.
        """
        if not self.config:
            return None
        
        # Roboflow / YOLO typically stores class names in 'names'
        names = self.config.get('names', [])
        
        # Sometimes names could be a dictionary in older YOLO formats, we handle both list and dict
        if isinstance(names, dict):
            # Sort by key to ensure correct order
            return [names[k] for k in sorted(names.keys())]
        elif isinstance(names, list):
            return names
            
        return None
        
    def get_num_classes(self) -> int:
        """
        Returns the number of classes defined in the dataset.
        """
        names = self.get_class_names()
        if names is not None:
            return len(names)
        return self.config.get('nc', 0)
