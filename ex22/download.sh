#!/bin/bash
# Script to download YOLOv5 repository and weights

echo "Setting up YOLOv5 environment..."

# Create weights directory if it doesn't exist
mkdir -p weights

# Clone YOLOv5 repository if not already downloaded
if [ ! -d "yolov5" ]; then
  echo "Cloning YOLOv5 repository..."
  git clone https://github.com/ultralytics/yolov5.git
  
  echo "Installing requirements..."
  pip install -r yolov5/requirements.txt
else
  echo "YOLOv5 repository already exists."
fi

# Download YOLOv5 weights if not already downloaded
if [ ! -f "weights/yolov5s.pt" ]; then
  echo "Downloading YOLOv5 weights..."
  curl -L https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -o weights/yolov5s.pt
else
  echo "YOLOv5 weights already exist."
fi

echo "Setup complete!"