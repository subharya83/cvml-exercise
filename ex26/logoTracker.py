import cv2
import numpy as np
import argparse
from sort import Sort  # Ensure you have the SORT algorithm implementation
from transformers import DetrForObjectDetection, DetrImageProcessor  # For DETR implementation
import torch

# Constants
ORB_METHOD = "orb"
DETR_METHOD = "detr"

# Initialize argument parser
parser = argparse.ArgumentParser(description='Track a logo in a video using an input image.')
parser.add_argument('--video', type=str, required=True, help='Path to the input video file.')
parser.add_argument('--image', type=str, required=True, help='Path to the input image file containing the logo.')
parser.add_argument('--output', type=str, required=True, help='Path to save the output video file.')
parser.add_argument('--method', type=str, default=ORB_METHOD, choices=[ORB_METHOD, DETR_METHOD],
                    help='Method to use for logo detection and tracking (orb or detr).')
args = parser.parse_args()

# Load the logo image
logo_image = cv2.imread(args.image, cv2.IMREAD_COLOR)
if logo_image is None:
    print("Error: Could not load logo image.")
    exit()
logo_height, logo_width = logo_image.shape[:2]


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


class DETRLogoDetector:
    """DETR-based logo detection and tracking."""

    def __init__(self, logo_image):
        self.logo_image = logo_image
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.logo_tensor = self.processor(images=logo_image, return_tensors="pt")["pixel_values"]

    def detect(self, frame):
        """Detect the logo in the frame using DETR."""
        inputs = self.processor(images=frame, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Process outputs to get bounding boxes
        logits = outputs.logits
        bboxes = outputs.pred_boxes
        # Filter based on confidence and class (customize as needed)
        # For simplicity, assume the logo is the first detected object
        if len(bboxes) > 0:
            x1, y1, x2, y2 = bboxes[0].tolist()
            return np.array([[x1, y1, x2, y2, 1.0]])  # Return detection with confidence 1.0
        return np.array([])  # No detection


def main():
    # Initialize detector based on method
    if args.method == ORB_METHOD:
        detector = ORBLogoDetector(logo_image)
    elif args.method == DETR_METHOD:
        detector = DETRLogoDetector(logo_image)
    else:
        raise ValueError(f"Invalid method: {args.method}")

    # Initialize SORT tracker
    mot_tracker = Sort()

    # Open the video
    cap = cv2.VideoCapture(args.video)
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

        # Display the resulting frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()