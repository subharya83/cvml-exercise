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