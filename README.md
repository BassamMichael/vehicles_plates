# YOLOv11 Object Detection Deployment

## Overview
This project provides a complete, production-ready scaffold for deploying a YOLOv11 object detection model. It is designed to be fully compatible with Streamlit Cloud, featuring a clean modular architecture and seamless integration with Roboflow YOLO dataset exports.

The implementation ensures robust handling of inferences, offering graceful fallback for class names and computing devices (CPU/GPU).

## Folder Structure

```
yolo-terminal-deployment/
├── app.py                  # Local CLI testing script
├── streamlit_app.py        # Streamlit web application entry point
├── requirements.txt        # Python dependencies
├── packages.txt            # System dependencies for Streamlit Cloud (libgl1)
├── README.md               # Documentation
│
├── model/                  # Model artifacts directory
│   ├── best.pt             # Trained YOLO weights (needs to be added)
│   └── labels.txt          # Fallback class labels (needs to be added)
│
├── dataset/                # Roboflow dataset directory
│   ├── images/             # Dataset images
│   ├── labels/             # Dataset YOLO labels
│   └── data.yaml           # YOLO dataset configuration
│
├── utils/                  # Core modules
│   ├── detector.py         # YOLO inference logic
│   ├── visualization.py    # OpenCV bounding box rendering
│   └── dataset_loader.py   # Dataset config and label parser
│
└── assets/                 # Static assets
    └── demo.png            # Demo image for initial testing
```

## Run Locally

First, install the required dependencies:
```bash
pip install -r requirements.txt
```

### Local Testing Script
Run the lightweight testing script to evaluate the model on the demo image:
```bash
python app.py
```
This will generate an `output.jpg` with the annotated results.

### Streamlit Web App
Launch the interactive web interface:
```bash
streamlit run streamlit_app.py
```

## Dataset Setup

1. Export your dataset from Roboflow in **YOLO format**.
2. Extract the downloaded zip file.
3. Place the `images/`, `labels/`, and `data.yaml` inside the `dataset/` folder of this project.

The system will automatically parse `dataset/data.yaml` to detect class names for visualization. If `data.yaml` is unavailable, it gracefully falls back to `model/labels.txt`, or the internal metadata of `model/best.pt`.

## Streamlit Cloud Deployment

1. **Commit your code to GitHub**: Ensure that all scripts, `requirements.txt`, and `packages.txt` are included.
2. **Add Model Weights**: Ensure `model/best.pt` is committed (use Git LFS if the file is large, or download it dynamically inside the script if preferred). The dataset folder is optional for deployment unless you plan to use it as a reference.
3. **Deploy on Streamlit Cloud**:
   - Link your GitHub repository.
   - Set the Main file path / Entry point to: `streamlit_app.py`
   - Click "Deploy". Streamlit will automatically install dependencies from `packages.txt` (like `libgl1` required for OpenCV) and `requirements.txt`.
