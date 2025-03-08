import cv2
import numpy as np
import argparse
from sort import Sort  # Ensure you have the SORT algorithm implementation

# Initialize argument parser
parser = argparse.ArgumentParser(description='Track a logo in a video using an input image.')
parser.add_argument('--video', type=str, required=True, help='Path to the input video file.')
parser.add_argument('--image', type=str, required=True, help='Path to the input image file containing the logo.')
parser.add_argument('--output', type=str, required=True, help='Path to save the output video file.')
args = parser.parse_args()

# Load the logo image
logo_image = cv2.imread(args.image, cv2.IMREAD_COLOR)
if logo_image is None:
    print("Error: Could not load logo image.")
    exit()
logo_height, logo_width = logo_image.shape[:2]

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

# Initialize feature matcher (ORB for logo detection)
orb = cv2.ORB_create()
logo_kp, logo_des = orb.detectAndCompute(logo_image, None)

# BFMatcher for matching logo features
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect ORB features in the current frame
    frame_kp, frame_des = orb.detectAndCompute(frame, None)

    if frame_des is not None and logo_des is not None:
        # Match features between the logo and the frame
        matches = bf.match(logo_des, frame_des)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract location of good matches
        src_pts = np.float32([logo_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography (projective transform) between logo and frame
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            # Transform the logo corners to the frame space
            h, w = logo_image.shape[:2]
            logo_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(logo_corners, M)

            # Get the bounding box from the transformed corners
            x_coords = [int(c[0][0]) for c in transformed_corners]
            y_coords = [int(c[0][1]) for c in transformed_corners]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)

            # Prepare detection for SORT (format: [x1, y1, x2, y2, confidence])
            detection = np.array([[x1, y1, x2, y2, 1.0]])  # Confidence is set to 1.0

            # Update SORT tracker
            trackers = mot_tracker.update(detection)

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

