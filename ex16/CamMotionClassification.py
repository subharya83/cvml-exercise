import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from scipy.linalg import logm, expm

class CamMotionClassifier:
    def __init__(self):
        self.surf = cv2.xfeatures2d.SURF_create()
        self.pca = PCA(n_components=2)
        self.svm = SVC(kernel='linear')

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

    def train(self, X, y):
        # Reduce dimensionality using PCA
        X_pca = self.pca.fit_transform(X)
        self.svm.fit(X_pca, y)

    def predict(self, X):
        X_pca = self.pca.transform(X)
        return self.svm.predict(X_pca)