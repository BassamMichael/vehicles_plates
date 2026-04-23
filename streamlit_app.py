import streamlit as st
from PIL import Image
import numpy as np
import os
from utils.detector import YOLOModel
from utils.visualization import draw_boxes

# Configure page
st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🔍",
    layout="centered"
)

# Initialize model (cache it to avoid reloading on every interaction)
@st.cache_resource
def load_detector():
    return YOLOModel()

detector = load_detector()

def main():
    st.title("🔍 YOLO Object Detection")
    st.markdown("Upload an image to detect objects or use the demo image.")

    # Confidence slider
    st.sidebar.header("Configuration")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.01,
        max_value=1.0,
        value=0.25,
        step=0.01,
        help="Filter out detections with confidence lower than this value."
    )

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png", "webp"]
    )
    
    # Track current image
    image = None
    
    # Load demo image if no file uploaded
    demo_path = "assets/demo.png"
    if uploaded_file is None:
        if os.path.exists(demo_path):
            st.info("Showing demo image. Upload an image to test on your own data.")
            try:
                image = Image.open(demo_path).convert('RGB')
            except Exception as e:
                st.error(f"Error loading demo image: {e}")
        else:
            st.info("Please upload an image to begin.")
    else:
        try:
            image = Image.open(uploaded_file).convert('RGB')
        except Exception as e:
            st.error(f"Invalid image format: {e}")
            return

    # Process image
    if image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        if st.button("🚀 Run Detection", type="primary"):
            if detector.model is None:
                st.error("Model could not be loaded. Please check if `model/best.pt` exists.")
            else:
                with st.spinner("Running inference..."):
                    try:
                        # Run detection
                        detections = detector.predict(image, conf_threshold=conf_threshold)
                        
                        # Draw boxes
                        annotated_img = draw_boxes(image, detections)
                        
                        with col2:
                            st.subheader("Detection Results")
                            st.image(annotated_img, use_container_width=True)
                            
                        # Show raw results
                        if detections:
                            st.success(f"Detected {len(detections)} objects!")
                            with st.expander("View Raw Results"):
                                st.json(detections)
                        else:
                            st.warning("No objects detected at this confidence threshold.")
                            
                    except Exception as e:
                        st.error(f"An error occurred during inference: {e}")

if __name__ == "__main__":
    main()
