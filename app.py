import os
import cv2
import numpy as np
from PIL import Image
from utils.detector import YOLOModel
from utils.visualization import draw_boxes

def main():
    print("Initializing YOLOv11 Detector...")
    detector = YOLOModel()
    
    if detector.model is None:
        print("Error: Model not found. Please ensure 'model/best.pt' exists.")
        return

    demo_image_path = "assets/demo.png"
    output_image_path = "output.jpg"

    if not os.path.exists(demo_image_path):
        print(f"Error: Demo image not found at {demo_image_path}.")
        # Create a dummy image for testing if not exists
        print("Creating a dummy image for testing purposes...")
        os.makedirs("assets", exist_ok=True)
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_img, "Dummy Image", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(demo_image_path, dummy_img)

    print(f"Loading image from {demo_image_path}...")
    try:
        image = Image.open(demo_image_path).convert('RGB')
    except Exception as e:
        print(f"Failed to load image: {e}")
        return

    print("Running inference...")
    detections = detector.predict(image, conf_threshold=0.25)
    
    print(f"Detected {len(detections)} objects:")
    for det in detections:
        print(f"- {det['class_name']}: {det['confidence']:.2f} at {det['box']}")

    print("Drawing bounding boxes...")
    annotated_np = draw_boxes(image, detections)
    
    # Convert RGB back to BGR for OpenCV saving
    annotated_bgr = cv2.cvtColor(annotated_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image_path, annotated_bgr)
    
    print(f"Successfully saved annotated image to {output_image_path}")

if __name__ == "__main__":
    main()
