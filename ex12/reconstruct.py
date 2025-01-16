import cv2
import numpy as np
import argparse
from pathlib import Path
import os
import logging

class PointCloudReconstructor:
    def __init__(self):
        self.images = []
        self.feature_detector = cv2.SIFT_create()
        self.matcher = cv2.FlannBasedMatcher_create()
        self.point_cloud = []  # List of (point, color) tuples
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_images(self, path):
        self.logger.info(f"Loading images from: {path}")
        path = Path(path)
        if path.is_file():  # Video file
            self.logger.info("Input is a video file")
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                self.logger.error("Failed to open video file")
                return False
                
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self.images.append(frame)
                frame_count += 1
                if frame_count % 10 == 0:  # Log every 10th frame
                    self.logger.info(f"Loaded {frame_count} frames")
            cap.release()
            self.logger.info(f"Completed video loading: extracted {len(self.images)} frames")
        else:  # Directory of images
            self.logger.info("Input is a directory")
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            for file in path.iterdir():
                if file.suffix.lower() in valid_extensions:
                    self.logger.debug(f"Loading image: {file.name}")
                    img = cv2.imread(str(file))
                    if img is not None:
                        self.images.append(img)
                    else:
                        self.logger.warning(f"Failed to load image: {file.name}")
            self.logger.info(f"Loaded {len(self.images)} images from directory")
                        
        return len(self.images) > 0
        
    def detect_and_match(self, img1, img2):
        self.logger.debug("Detecting and matching features between image pair")
        # Detect keypoints and compute descriptors
        kp1, des1 = self.feature_detector.detectAndCompute(img1, None)
        kp2, des2 = self.feature_detector.detectAndCompute(img2, None)
        
        self.logger.debug(f"Found {len(kp1)} keypoints in first image and {len(kp2)} in second image")
        
        # Match features
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        pts1 = []
        pts2 = []
        
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)
        
        self.logger.debug(f"Found {len(good_matches)} good matches after ratio test")
                
        return np.float32(pts1), np.float32(pts2)
        
    def triangulate_points(self, img1, img2, R, t, pts1, pts2):
        self.logger.debug("Beginning point triangulation")
        # Camera matrix (assuming default parameters)
        focal_length = max(img1.shape[1], img1.shape[0])
        pp = (img1.shape[1]//2, img1.shape[0]//2)
        K = np.array([
            [focal_length, 0, pp[0]],
            [0, focal_length, pp[1]],
            [0, 0, 1]
        ])
        
        # Projection matrices
        P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
        P2 = K @ np.hstack((R, t))
        
        # Triangulate
        points4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points4D /= points4D[3]
        points3D = points4D[:3].T
        
        initial_point_count = len(self.point_cloud)
        # Get colors from first image
        for i, (pt3d, pt2d) in enumerate(zip(points3D, pts1)):
            x, y = int(pt2d[0]), int(pt2d[1])
            if 0 <= x < img1.shape[1] and 0 <= y < img1.shape[0]:
                color = img1[y, x]
                self.point_cloud.append((pt3d, color))
        
        new_points = len(self.point_cloud) - initial_point_count
        self.logger.debug(f"Added {new_points} new 3D points to cloud")
                
    def save_ply(self, filename):
        self.logger.info(f"Saving point cloud to PLY file: {filename}")
        self.logger.info(f"Total points in cloud: {len(self.point_cloud)}")
        
        try:
            with open(filename, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(self.point_cloud)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")
                
                for point, color in self.point_cloud:
                    f.write(f"{point[0]} {point[1]} {point[2]} "
                           f"{int(color[2])} {int(color[1])} {int(color[0])}\n")
            self.logger.info("Successfully saved PLY file")
        except Exception as e:
            self.logger.error(f"Failed to save PLY file: {str(e)}")
            raise
                
    def reconstruct(self, input_path, output_path):
        self.logger.info("Starting 3D reconstruction")
        if not self.load_images(input_path):
            self.logger.error("Failed to load images")
            return False
            
        self.logger.info(f"Processing {len(self.images)} images")
        for i in range(len(self.images) - 1):
            self.logger.info(f"Processing image pair {i+1}/{len(self.images)-1}")
            pts1, pts2 = self.detect_and_match(self.images[i], self.images[i+1])
            
            if len(pts1) < 8:
                self.logger.warning(f"Insufficient matching points ({len(pts1)}) for image pair {i+1}, skipping")
                continue
                
            # Find essential matrix and recover pose
            E, mask = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(0,0))
            _, R, t, mask = cv2.recoverPose(E, pts1, pts2)
            
            # Triangulate points
            self.triangulate_points(self.images[i], self.images[i+1], R, t, pts1, pts2)
            
        if not self.point_cloud:
            self.logger.error("Failed to reconstruct point cloud - no points generated")
            return False
            
        self.save_ply(output_path)
        self.logger.info("Reconstruction completed successfully")
        return True

def main():
    parser = argparse.ArgumentParser(description='3D reconstruction from images/video')
    parser.add_argument('-i', required=True, help='Input video file or image directory')
    parser.add_argument('-o', required=True, help='Output PLY file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    reconstructor = PointCloudReconstructor()
    try:
        if reconstructor.reconstruct(args.i, args.o):
            logging.info("Reconstruction completed successfully")
        else:
            logging.error("Reconstruction failed")
    except Exception as e:
        logging.error(f"Reconstruction failed with error: {str(e)}")
        raise

if __name__ == '__main__':
    main()
