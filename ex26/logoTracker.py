import cv2
import numpy as np
import argparse
from sort import Sort  # Ensure you have the SORT algorithm implementation
import torch
import onnxruntime as ort  # For ONNX inference


class ORBLogoDetector:
    """ORB-based logo detection and tracking."""

    def __init__(self, logo_image):
        self.logo_image = logo_image
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.logo_kp, self.logo_des = self.orb.detectAndCompute(logo_image, None)

    def detect(self, frame):
        """Detect the logo in the frame using ORB feature matching."""
        frame_kp, frame_des = self.orb.detectAndCompute(frame, None)
        if frame_des is not None and self.logo_des is not None:
            matches = self.bf.match(self.logo_des, frame_des)
            matches = sorted(matches, key=lambda x: x.distance)

            src_pts = np.float32([self.logo_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                h, w = self.logo_image.shape[:2]
                logo_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                transformed_corners = cv2.perspectiveTransform(logo_corners, M)
                x_coords = [int(c[0][0]) for c in transformed_corners]
                y_coords = [int(c[0][1]) for c in transformed_corners]
                x1, y1 = min(x_coords), min(y_coords)
                x2, y2 = max(x_coords), max(y_coords)
                return np.array([[x1, y1, x2, y2, 1.0]])  # Return detection with confidence 1.0
        return np.array([])  # No detection


class LightweightDETRLogoDetector:
    """Lightweight DETR-based logo detection and tracking."""

    def __init__(self, model_path):
        # Initialize ONNX runtime session
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame):
        """Detect the logo in the frame using a lightweight DETR model."""
        # Preprocess the frame
        input_tensor = self.preprocess(frame)

        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})

        # Postprocess outputs to get bounding boxes
        bboxes = self.postprocess(outputs)
        if len(bboxes) > 0:
            return np.array([[bboxes[0][0], bboxes[0][1], bboxes[0][2], bboxes[0][3], 1.0]])  # Return detection with confidence 1.0
        return np.array([])  # No detection

    def preprocess(self, frame):
        """Preprocess the frame for DETR input."""
        # Resize and normalize the frame
        frame = cv2.resize(frame, (800, 800))  # DETR expects 800x800 input
        frame = frame / 255.0  # Normalize to [0, 1]
        frame = np.transpose(frame, (2, 0, 1))  # Change to CHW format
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension
        frame = frame.astype(np.float32)
        return frame

    def postprocess(self, outputs):
        """Postprocess DETR outputs to extract bounding boxes."""
        # Extract logits and bounding boxes from outputs
        logits = outputs[0]  # Assume logits are the first output
        bboxes = outputs[1]  # Assume bounding boxes are the second output

        # Filter based on confidence threshold (customize as needed)
        confidence_threshold = 0.5
        if len(logits) > 0 and len(bboxes) > 0:
            # For simplicity, return the first detected object
            return bboxes[0]
        return []


def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Track a logo in a video using an input image.')
    parser.add_argument('-v', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('-i', type=str, required=True, help='Path to the input image file containing the logo.')
    parser.add_argument('-o', type=str, required=True, help='Path to save the output video file.')
    parser.add_argument('-a', type=str, default=0, choices=[0, 1],
                        help='Logo detection/tracking algorithm [orb:0|detr:1].')
    args = parser.parse_args()

    # Load the logo image
    logo_image = cv2.imread(args.i, cv2.IMREAD_COLOR)
    if logo_image is None:
        print("Error: Could not load logo image.")
        exit()
    logo_height, logo_width = logo_image.shape[:2]
    # Initialize detector based on method
    if int(args.a) == 0:
        detector = ORBLogoDetector(logo_image)
    elif int(args.a) == 1:
        # Load lightweight DETR model
        model_path = "weights/detr_resnet50.onnx"  # Path to locally saved ONNX model
        detector = LightweightDETRLogoDetector(model_path)
    else:
        raise ValueError(f"Invalid method: {args.a}")

    # Initialize SORT tracker
    mot_tracker = Sort()

    # Open the video
    cap = cv2.VideoCapture(args.v)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.output, fourcc, 20.0, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect the logo
        detections = detector.detect(frame)

        # Update SORT tracker
        if len(detections) > 0:
            trackers = mot_tracker.update(detections)

            # Draw tracked bounding box
            for track in trackers:
                x1, y1, x2, y2, track_id = map(int, track)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write the frame into the output video file
        out.write(frame)

    # Release everything if job is finished
    cap.release()
    out.release()


if __name__ == "__main__":
    main()
