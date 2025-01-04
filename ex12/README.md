I'll help you create code for 3D reconstruction from video/images using OpenCV. This will use Structure from Motion (SfM) techniques to create a point cloud.



```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

class PointCloudReconstructor {
private:
    std::vector<cv::Mat> images;
    cv::Ptr<cv::Feature2D> feature_detector;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    
    struct CloudPoint {
        cv::Point3d pt;
        cv::Vec3b color;
    };
    
    std::vector<CloudPoint> point_cloud;

    bool loadImages(const std::string& path) {
        if (fs::is_regular_file(path)) {  // Video file
            cv::VideoCapture cap(path);
            if (!cap.isOpened()) return false;
            
            cv::Mat frame;
            while (cap.read(frame)) {
                images.push_back(frame.clone());
            }
        } else {  // Directory of images
            for (const auto& entry : fs::directory_iterator(path)) {
                cv::Mat img = cv::imread(entry.path().string());
                if (!img.empty()) {
                    images.push_back(img);
                }
            }
        }
        return !images.empty();
    }

    void detectAndMatch(const cv::Mat& img1, const cv::Mat& img2,
                       std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2) {
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        cv::Mat descriptors1, descriptors2;
        
        feature_detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
        feature_detector->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);
        
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
        
        // Ratio test
        std::vector<cv::DMatch> good_matches;
        for (const auto& match : knn_matches) {
            if (match[0].distance < 0.7 * match[1].distance) {
                good_matches.push_back(match[0]);
            }
        }
        
        pts1.clear();
        pts2.clear();
        for (const auto& match : good_matches) {
            pts1.push_back(keypoints1[match.queryIdx].pt);
            pts2.push_back(keypoints2[match.trainIdx].pt);
        }
    }

    void triangulatePoints(const cv::Mat& img1, const cv::Mat& img2,
                          const cv::Mat& R, const cv::Mat& t,
                          const std::vector<cv::Point2f>& pts1,
                          const std::vector<cv::Point2f>& pts2) {
        // Camera matrix (assuming default parameters)
        double focal_length = std::max(img1.cols, img1.rows);
        cv::Point2d pp(img1.cols/2, img1.rows/2);
        cv::Mat K = (cv::Mat_<double>(3,3) << 
            focal_length, 0, pp.x,
            0, focal_length, pp.y,
            0, 0, 1);
        
        // Projection matrices
        cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_64F);
        cv::Mat P2 = K * cv::hconcat(R, t);
        
        cv::Mat points4D;
        cv::triangulatePoints(P1, P2, pts1, pts2, points4D);
        
        // Convert to 3D points
        for (int i = 0; i < points4D.cols; i++) {
            cv::Mat x = points4D.col(i);
            x /= x.at<float>(3);
            CloudPoint cp;
            cp.pt = cv::Point3d(x.at<float>(0), x.at<float>(1), x.at<float>(2));
            
            // Get color from first image
            cv::Point2f pt = pts1[i];
            if (pt.x >= 0 && pt.x < img1.cols && pt.y >= 0 && pt.y < img1.rows) {
                cp.color = img1.at<cv::Vec3b>(pt);
            }
            point_cloud.push_back(cp);
        }
    }

    void savePLY(const std::string& filename) {
        std::ofstream out(filename);
        out << "ply\n";
        out << "format ascii 1.0\n";
        out << "element vertex " << point_cloud.size() << "\n";
        out << "property float x\n";
        out << "property float y\n";
        out << "property float z\n";
        out << "property uchar red\n";
        out << "property uchar green\n";
        out << "property uchar blue\n";
        out << "end_header\n";
        
        for (const auto& point : point_cloud) {
            out << point.pt.x << " " << point.pt.y << " " << point.pt.z << " "
                << (int)point.color[2] << " " << (int)point.color[1] << " " << (int)point.color[0] << "\n";
        }
    }

public:
    PointCloudReconstructor() {
        feature_detector = cv::SIFT::create();
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }
    
    bool reconstruct(const std::string& input_path, const std::string& output_path) {
        if (!loadImages(input_path)) {
            std::cerr << "Failed to load images\n";
            return false;
        }
        
        for (size_t i = 0; i < images.size() - 1; i++) {
            std::vector<cv::Point2f> pts1, pts2;
            detectAndMatch(images[i], images[i+1], pts1, pts2);
            
            if (pts1.size() < 8) continue;
            
            cv::Mat E = cv::findEssentialMat(pts1, pts2, 1.0, cv::Point2d(0,0));
            cv::Mat R, t;
            cv::recoverPose(E, pts1, pts2, R, t);
            
            triangulatePoints(images[i], images[i+1], R, t, pts1, pts2);
        }
        
        if (point_cloud.empty()) {
            std::cerr << "Failed to reconstruct point cloud\n";
            return false;
        }
        
        savePLY(output_path);
        return true;
    }
};

int main(int argc, char** argv) {
    std::string input_path, output_path;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-i" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            output_path = argv[++i];
        }
    }
    
    if (input_path.empty() || output_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " -i <input_path> -o <output_path>\n";
        return 1;
    }
    
    PointCloudReconstructor reconstructor;
    if (!reconstructor.reconstruct(input_path, output_path)) {
        std::cerr << "Reconstruction failed\n";
        return 1;
    }
    
    std::cout << "Reconstruction completed successfully\n";
    return 0;
}

```

```python
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
    parser.add_argument('-i', '--input', required=True, help='Input video file or image directory')
    parser.add_argument('-o', '--output', required=True, help='Output PLY file')
    args = parser.parse_args()
    
    reconstructor = PointCloudReconstructor()
    if reconstructor.reconstruct(args.input, args.output):
        print("Reconstruction completed successfully")
    else:
        print("Reconstruction failed")

if __name__ == '__main__':
    main()

```

I've created both C++ and Python implementations of a 3D reconstruction pipeline using OpenCV. Both versions implement the following features:

1. Support for both video files and image directories as input
2. Feature detection and matching using SIFT
3. Essential matrix estimation and pose recovery
4. Point triangulation
5. Color information preservation
6. PLY file output format with both geometry and color data

Key components of the implementation:

1. Feature Detection and Matching:
   - Uses SIFT features for robust point detection
   - Implements ratio test for match filtering
   - FLANN-based matcher for efficient matching

2. 3D Reconstruction:
   - Essential matrix estimation for relative pose
   - Pose recovery using decomposition
   - Point triangulation using the recovered poses
   - Color sampling from original images

3. File Handling:
   - Supports both video and image directory inputs
   - Saves output as PLY format with color information

To use either version:

```bash
# C++ version
./reconstruct -i input_video.mp4 -o output.ply
# or
./reconstruct -i ./images_directory -o output.ply

# Python version
python reconstruct.py -i input_video.mp4 -o output.ply

