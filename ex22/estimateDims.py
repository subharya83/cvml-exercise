import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
import subprocess
import argparse
from pathlib import Path

class DimensionEstimator:
    def __init__(self):
        # Known dimensions of a standard soda can in cm
        self.soda_can_height = 12.2  # standard height in cm
        self.soda_can_diameter = 6.6  # standard diameter in cm
        
        # Run the bash script to download YOLOv5
        self._setup_yolo()
        
        # Import YOLOv5 modules (after they're downloaded)
        import sys
        sys.path.append("yolov5")
        from models.experimental import attempt_load
        from utils.general import non_max_suppression, scale_coords
        from utils.torch_utils import select_device
        from utils.datasets import letterbox
        
        # These imports are used in the methods below
        self.non_max_suppression = non_max_suppression
        self.scale_coords = scale_coords
        self.letterbox = letterbox
        
        # Load YOLOv5 model
        self.device = select_device('')
        self.model = attempt_load("weights/yolov5s.pt", device=self.device)
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
    
    def _setup_yolo(self):
        """Run the bash script to download YOLOv5 and weights"""
        try:
            # Make the script executable
            script_path = "download_yolo.sh"
            os.chmod(script_path, 0o755)
            
            # Run the bash script
            print("Running YOLOv5 setup script...")
            result = subprocess.run(
                [f"./{script_path}"], 
                shell=True, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            print(result.stdout)
            
            if result.returncode != 0:
                print(f"Error running setup script: {result.stderr}")
                raise Exception("Failed to set up YOLOv5")
                
        except Exception as e:
            print(f"Error setting up YOLOv5: {str(e)}")
            raise
        
    def preprocess_image(self, img_path, img_size=640):
        """Preprocess the image for YOLOv5"""
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize and pad image
        img_processed = self.letterbox(img, img_size, stride=32)[0]
        
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
        pred = self.non_max_suppression(pred, conf_thres, iou_thres)
        return pred
    
    def estimate_dimensions(self, img_path, output_path=None, target_type="tree"):
        """
        Estimate dimensions of a building or tree using a soda can as reference
        
        Args:
            img_path: Path to the image
            output_path: Path to save the output image with visualization
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
            pred[:, :4] = self.scale_coords(img_processed.shape[2:], pred[:, :4], original_img.shape).round()
            
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
            
            # Draw the results on the image and save if output path is provided
            if output_path:
                self._visualize_results(original_img, soda_can_bbox, target_bbox, 
                                      estimated_height_m, estimated_width_m, output_path)
            
            return {
                "estimated_height_m": estimated_height_m,
                "estimated_width_m": estimated_width_m,
                "reference_object": "soda can",
                "reference_height_cm": self.soda_can_height,
                "success": True
            }
        else:
            missing = []
            if not soda_can_bbox:
                missing.append("soda can")
            if not target_bbox:
                missing.append(target_type)
                
            return {
                "error": f"Could not detect {' and '.join(missing)} in the image.",
                "detected_objects": [self.class_names[int(cls)] for *_, cls in predictions[0]],
                "success": False
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
    
    def _visualize_results(self, img, can_bbox, target_bbox, height_m, width_m, output_path):
        """Draw bounding boxes and dimensions on the image and save to output path"""
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
        result_img.save(output_path)
        print(f"Result saved as {output_path}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Estimate dimensions of a building or tree from a single image.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input image")
    parser.add_argument("-o", "--output", help="Path to save the output image with dimension visualization")
    parser.add_argument("-t", "--type", choices=["tree", "building"], default="tree", 
                        help="Type of object to measure (tree or building)")
    
    return parser.parse_args()

def main():
    """Main function to run the dimension estimator"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize the dimension estimator
    print("Initializing dimension estimator...")
    estimator = DimensionEstimator()
    
    # Set default output path if not specified
    output_path = args.output
    if not output_path:
        # Use input filename with _result suffix
        input_path = Path(args.input)
        output_path = str(input_path.with_stem(f"{input_path.stem}_result"))
    
    # Estimate dimensions
    print(f"Processing image: {args.input}")
    print(f"Target type: {args.type}")
    results = estimator.estimate_dimensions(args.input, output_path, args.type)
    
    # Print results
    if results["success"]:
        print(f"\nEstimated dimensions:")
        print(f"Height: {results['estimated_height_m']:.2f} meters")
        print(f"Width: {results['estimated_width_m']:.2f} meters")
        print(f"Reference object: {results['reference_object']} " +
              f"(height: {results['reference_height_cm']} cm)")
        print(f"\nVisualization saved to: {output_path}")
    else:
        print(f"Error: {results['error']}")
        print(f"Detected objects in the image: {results['detected_objects']}")

if __name__ == "__main__":
    main()
    