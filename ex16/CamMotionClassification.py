import cv2
import numpy as np
from scipy.linalg import logm
import os

class CamMotionClassifier:
    def __init__(self):
        self.surf = cv2.xfeatures2d.SURF_create()

    def compute_homography(self, frame1, frame2):
        # Detect SURF features
        kp1, des1 = self.surf.detectAndCompute(frame1, None)
        kp2, des2 = self.surf.detectAndCompute(frame2, None)

        # Match features using FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Filter matches using Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Compute homography if enough matches are found
        if len(good_matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return H
        else:
            return None

    def lie_algebra_mapping(self, H):
        # Map homography to Lie algebra space
        M = logm(H)
        return M.flatten()

    def extract_features(self, video_path, frame_skip=4):
        cap = cv2.VideoCapture(video_path)
        features = []
        prev_frame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if prev_frame is not None:
                H = self.compute_homography(prev_frame, frame)
                if H is not None:
                    M = self.lie_algebra_mapping(H)
                    features.append(M)

            prev_frame = frame
            for _ in range(frame_skip):
                cap.read()  # Skip frames

        cap.release()
        return np.array(features)

    def create_dataset(self, video_dir, output_file):
        X = []
        y = []

        # Iterate through all video files in the directory
        for video_file in os.listdir(video_dir):
            if video_file.endswith(".mp4"):
                video_path = os.path.join(video_dir, video_file)
                shot_name = video_file.split('_')[0]  # Extract shot name from filename

                # Extract features from the video
                features = self.extract_features(video_path)
                if len(features) > 0:
                    X.append(np.mean(features, axis=0))  # Use mean feature vector
                    y.append(shot_name)

        # Save the dataset to a file
        np.savez(output_file, X=np.array(X), y=np.array(y))
        print(f"Dataset created and saved to {output_file}")