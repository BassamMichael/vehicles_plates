import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Union

def draw_boxes(image: Union[np.ndarray, Image.Image], detections: List[Dict[str, Any]]) -> np.ndarray:
    """
    Draws bounding boxes and labels on an image based on detections.
    
    Args:
        image: PIL Image or numpy array (RGB/BGR).
        detections: List of detection dictionaries.
        
    Returns:
        Annotated numpy array image (RGB format).
    """
    # Convert PIL Image to numpy array if necessary
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image.copy()
        
    # Ensure image is in RGB for proper color drawing (assuming Streamlit/Matplotlib context)
    # If it was loaded with cv2.imread it would be BGR, but PIL/Streamlit gives RGB.
    # We will assume img_np is RGB.

    img_h, img_w = img_np.shape[:2]
    
    # Adaptive sizing based on image dimensions
    thickness = max(1, int(min(img_w, img_h) / 300))
    font_scale = max(0.5, min(img_w, img_h) / 800)
    
    # Color palette for classes (using a simple hash to assign colors)
    colors = {}
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['box'])
        conf = det['confidence']
        cls_id = det['class_id']
        cls_name = det['class_name']
        
        # Generate a distinct color for each class based on ID
        if cls_id not in colors:
            np.random.seed(cls_id)
            # Generate bright RGB colors
            colors[cls_id] = tuple(int(c) for c in np.random.randint(50, 255, size=3))
            
        color = colors[cls_id]
        
        # Draw bounding box
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label background
        label = f"{cls_name} {conf:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness-1)
        )
        
        # Ensure label stays within image bounds
        y1_label = max(y1, text_height + 5)
        
        cv2.rectangle(
            img_np, 
            (x1, y1_label - text_height - 5), 
            (x1 + text_width, y1_label + baseline - 5), 
            color, 
            -1
        )
        
        # Draw text (white or black depending on background brightness)
        # Assuming bright background, use dark text or vice versa. We'll use white for simplicity
        # with dark text fallback if needed, but mostly white looks good on random bright colors.
        text_color = (255, 255, 255)
        if sum(color) > 500: # If color is very bright
            text_color = (0, 0, 0)
            
        cv2.putText(
            img_np, 
            label, 
            (x1, y1_label - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            text_color, 
            max(1, thickness-1), 
            cv2.LINE_AA
        )
        
    return img_np
