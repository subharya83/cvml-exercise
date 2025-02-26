import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import requests
from pathlib import Path

# Create weights directory if it doesn't exist
os.makedirs("weights", exist_ok=True)

# Download YOLOv5 for object detection if not already downloaded
if not os.path.exists("yolov5"):
    print("Cloning YOLOv5 repository...")
    os.system("git clone https://github.com/ultralytics/yolov5.git")
    os.system("pip install -r yolov5/requirements.txt")

# Import YOLOv5 modules
import sys
sys.path.append("yolov5")
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.datasets import letterbox

# Download YOLOv5 weights if not already downloaded
weights_path = "weights/yolov5s.pt"
if not os.path.exists(weights_path):
    print(f"Downloading YOLOv5 weights to {weights_path}...")
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    torch.hub.download_url_to_file(
        "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt",
        weights_path
    )

class DimensionEstimator:
    def __init__(self):
        # Known dimensions of a standard soda can in cm
        self.soda_can_height = 12.2  # standard height in cm
        self.soda_can_diameter = 6.6  # standard diameter in cm
        
        # Load YOLOv5 model
        self.device = select_device('')
        self.model = attempt_load(weights_path, device=self.device)
        self.model.eval()
        
        # Classes that YOLO can detect (COCO dataset)
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
                           'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
                           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
                           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
                           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
                           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
                           'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
                           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
                           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
                           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
                           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
                           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
                           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
                           'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        
        # The bottle class (index 39) will be used to detect soda can
        self.bottle_class_id = 39
        
    def preprocess_image(self, img_path, img_size=640):
        """Preprocess the image for YOLOv5"""
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize and pad image
        img_processed = letterbox(img, img_size, stride=32)[0]
        
        # Convert to float and normalize
        img_processed = img_processed.transpose(2, 0, 1)  # HWC to CHW
        img_processed = np.ascontiguousarray(img_processed)
        img_processed = torch.from_numpy(img_processed).to(self.device)
        img_processed = img_processed.float() / 255.0
        
        if img_processed.ndimension() == 3:
            img_processed = img_processed.unsqueeze(0)
            
        return img, img_processed
        
    def detect_objects(self, img_processed, conf_thres=0.25, iou_thres=0.45):
        """Run YOLOv5 inference on the image"""
        with torch.no_grad():
            pred = self.model(img_processed)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        return pred
    
    def estimate_dimensions(self, img_path, target_type="tree"):
        """
        Estimate dimensions of a building or tree using a soda can as reference
        
        Args:
            img_path: Path to the image
            target_type: Either "tree" or "building" to determine what to look for
            
        Returns:
            Estimated height and width in meters
        """
        # Process image
        original_img, img_processed = self.preprocess_image(img_path)
        orig_height, orig_width = original_img.shape[:2]
        
        # Run detection
        predictions = self.detect_objects(img_processed)
        
        # Process results
        soda_can_bbox = None
        target_bbox = None
        
        if len(predictions[0]) > 0:
            # Scale boxes to original image
            pred = predictions[0]
            pred[:, :4] = scale_coords(img_processed.shape[2:], pred[:, :4], original_img.shape).round()
            
            # Look for soda can (bottle class in COCO)
            for *xyxy, conf, cls_id in pred:
                if int(cls_id) == self.bottle_class_id and conf > 0.5:
                    soda_can_bbox = [int(coord) for coord in xyxy]
                
                # For tree, look for potted plant or detect using color/texture
                if target_type == "tree" and int(cls_id) == 58:  # potted plant
                    target_bbox = [int(coord) for coord in xyxy]
                    
            # If the model couldn't detect the tree or building, we need a different approach
            if target_bbox is None:
                if target_type == "tree":
                    # Simplified tree detection using color segmentation
                    target_bbox = self._detect_tree(original_img)
                elif target_type == "building":
                    # Simplified building detection using edge detection
                    target_bbox = self._detect_building(original_img)
        
        # If we have both the soda can and the target object
        if soda_can_bbox and target_bbox:
            # Calculate pixel dimensions
            can_height_px = soda_can_bbox[3] - soda_can_bbox[1]
            target_height_px = target_bbox[3] - target_bbox[1]
            target_width_px = target_bbox[2] - target_bbox[0]
            
            # Calculate scaling factor based on known can height
            scale_factor = self.soda_can_height / can_height_px
            
            # Estimate dimensions in cm, convert to meters
            estimated_height_m = (target_height_px * scale_factor) / 100
            estimated_width_m = (target_width_px * scale_factor) / 100
            
            # Draw the results on the image and save
            self._visualize_results(original_img, soda_can_bbox, target_bbox, 
                                   estimated_height_m, estimated_width_m)
            
            return {
                "estimated_height_m": estimated_height_m,
                "estimated_width_m": estimated_width_m,
                "reference_object": "soda can",
                "reference_height_cm": self.soda_can_height
            }
        else:
            missing = []
            if not soda_can_bbox:
                missing.append("soda can")
            if not target_bbox:
                missing.append(target_type)
                
            return {
                "error": f"Could not detect {' and '.join(missing)} in the image.",
                "detected_objects": [self.class_names[int(cls)] for *_, cls in predictions[0]]
            }
    
    def _detect_tree(self, img):
        """Basic tree detection using color segmentation"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Define green color range for trees
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([90, 255, 255])
        
        # Create mask for green areas
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (assuming it's the tree)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return [x, y, x + w, y + h]
        
        return None
    
    def _detect_building(self, img):
        """Basic building detection using edge detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours in the edge map
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contours
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Look for rectangular shapes (typical for buildings)
            for contour in contours[:5]:  # Check the 5 largest contours
                # Approximate the contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # If it has 4-6 sides, it might be a building
                if 4 <= len(approx) <= 6:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Filter out small detections
                    if w > img.shape[1] / 10 and h > img.shape[0] / 10:
                        return [x, y, x + w, y + h]
        
        # If no suitable contour found, return the whole image as fallback
        h, w = img.shape[:2]
        return [0, 0, w, h]
    
    def _visualize_results(self, img, can_bbox, target_bbox, height_m, width_m):
        """Draw bounding boxes and dimensions on the image"""
        result_img = Image.fromarray(img.copy())
        draw = ImageDraw.Draw(result_img)
        
        # Draw soda can bbox
        draw.rectangle(
            [(can_bbox[0], can_bbox[1]), (can_bbox[2], can_bbox[3])],
            outline="blue",
            width=2
        )
        draw.text((can_bbox[0], can_bbox[1] - 10), "Soda Can (Reference)", fill="blue")
        
        # Draw target bbox
        draw.rectangle(
            [(target_bbox[0], target_bbox[1]), (target_bbox[2], target_bbox[3])],
            outline="red",
            width=2
        )
        draw.text(
            (target_bbox[0], target_bbox[1] - 30),
            f"Height: {height_m:.2f}m, Width: {width_m:.2f}m",
            fill="red"
        )
        
        # Save the result
        result_img.save("dimension_estimation_result.jpg")
        print(f"Result saved as dimension_estimation_result.jpg")
        
        # Display the result
        plt.figure(figsize=(10, 8))
        plt.imshow(result_img)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    estimator = DimensionEstimator()
    
    # Get input from user
    img_path = input("Enter path to image file: ")
    target_type = input("What are you measuring? (tree/building): ").lower()
    
    if target_type not in ["tree", "building"]:
        print("Invalid target type. Defaulting to 'tree'.")
        target_type = "tree"
    
    # Estimate dimensions
    results = estimator.estimate_dimensions(img_path, target_type)
    
    # Print results
    if "error" in results:
        print(f"Error: {results['error']}")
        print(f"Detected objects in the image: {results['detected_objects']}")
    else:
        print(f"\nEstimated dimensions:")
        print(f"Height: {results['estimated_height_m']:.2f} meters")
        print(f"Width: {results['estimated_width_m']:.2f} meters")
        print(f"Reference object: {results['reference_object']} " +
              f"(height: {results['reference_height_cm']} cm)")
        print("\nResult image saved as dimension_estimation_result.jpg")