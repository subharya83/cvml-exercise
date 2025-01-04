import cv2
import numpy as np
import argparse
from pathlib import Path
import os

class PointCloudReconstructor:
    def __init__(self):
        self.images = []
        self.feature_detector = cv2.SIFT_create()
        self.matcher = cv2.FlannBasedMatcher_create()
        self.point_cloud = []  # List of (point, color) tuples
        
    def load_images(self, path):
        path = Path(path)
        if path.is_file():  # Video file
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return False
                
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self.images.append(frame)
            cap.release()
        else:  # Directory of images
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            for file in path.iterdir():
                if file.suffix.lower() in valid_extensions:
                    img = cv2.imread(str(file))
                    if img is not None:
                        self.images.append(img)
                        
        return len(self.images) > 0
        
    def detect_and_match(self, img1, img2):
        # Detect keypoints and compute descriptors
        kp1, des1 = self.feature_detector.detectAndCompute(img1, None)
        kp2, des2 = self.feature_detector.detectAndCompute(img2, None)
        
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
                
        return np.float32(pts1), np.float32(pts2)
        
    def triangulate_points(self, img1, img2, R, t, pts1, pts2):
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
        
        # Get colors from first image
        for i, (pt3d, pt2d) in enumerate(zip(points3D, pts1)):
            x, y = int(pt2d[0]), int(pt2d[1])
            if 0 <= x < img1.shape[1] and 0 <= y < img1.shape[0]:
                color = img1[y, x]
                self.point_cloud.append((pt3d, color))
                
    def save_ply(self, filename):
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
                
    def reconstruct(self, input_path, output_path):
        if not self.load_images(input_path):
            print("Failed to load images")
            return False
            
        for i in range(len(self.images) - 1):
            pts1, pts2 = self.detect_and_match(self.images[i], self.images[i+1])
            
            if len(pts1) < 8:
                continue
                
            # Find essential matrix and recover pose
            E, mask = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(0,0))
            _, R, t, mask = cv2.recoverPose(E, pts1, pts2)
            
            # Triangulate points
            self.triangulate_points(self.images[i], self.images[i+1], R, t, pts1, pts2)
            
        if not self.point_cloud:
            print("Failed to reconstruct point cloud")
            return False
            
        self.save_ply(output_path)
        return True

def main():
    parser = argparse.ArgumentParser(description='3D reconstruction from images/video')
    parser.add_argument('-i', required=True, help='Input video file or image directory')
    parser.add_argument('-o',  required=True, help='Output PLY file')
    args = parser.parse_args()
    
    reconstructor = PointCloudReconstructor()
    if reconstructor.reconstruct(args.i, args.o):
        print("Reconstruction completed successfully")
    else:
        print("Reconstruction failed")

if __name__ == '__main__':
    main()