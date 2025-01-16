#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

// Simple logging utility
class Logger {
public:
    enum Level { DEBUG, INFO, WARNING, ERROR };
    
    static void setLevel(Level level) { currentLevel = level; }
    
    template<typename... Args>
    static void debug(Args... args) { log(DEBUG, args...); }
    
    template<typename... Args>
    static void info(Args... args) { log(INFO, args...); }
    
    template<typename... Args>
    static void warning(Args... args) { log(WARNING, args...); }
    
    template<typename... Args>
    static void error(Args... args) { log(ERROR, args...); }

private:
    static Level currentLevel;
    
    static std::string getTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
    
    static const char* getLevelString(Level level) {
        switch (level) {
            case DEBUG: return "DEBUG";
            case INFO: return "INFO";
            case WARNING: return "WARNING";
            case ERROR: return "ERROR";
            default: return "UNKNOWN";
        }
    }
    
    template<typename T, typename... Args>
    static void log(Level level, T first, Args... args) {
        if (level >= currentLevel) {
            std::cout << getTimestamp() << " - " << getLevelString(level) << " - " 
                     << first;
            ((std::cout << " " << args), ...);
            std::cout << std::endl;
        }
    }
};

Logger::Level Logger::currentLevel = Logger::INFO;

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
        Logger::info("Loading images from:", path);
        
        if (fs::is_regular_file(path)) {
            Logger::info("Input is a video file");
            cv::VideoCapture cap(path);
            if (!cap.isOpened()) {
                Logger::error("Failed to open video file");
                return false;
            }
            
            cv::Mat frame;
            int frameCount = 0;
            while (cap.read(frame)) {
                images.push_back(frame.clone());
                frameCount++;
                if (frameCount % 10 == 0) {
                    Logger::info("Loaded", frameCount, "frames");
                }
            }
            Logger::info("Completed video loading: extracted", images.size(), "frames");
        } else {
            Logger::info("Input is a directory");
            int loadedCount = 0;
            for (const auto& entry : fs::directory_iterator(path)) {
                Logger::debug("Loading image:", entry.path().string());
                cv::Mat img = cv::imread(entry.path().string());
                if (!img.empty()) {
                    images.push_back(img);
                    loadedCount++;
                } else {
                    Logger::warning("Failed to load image:", entry.path().string());
                }
            }
            Logger::info("Loaded", loadedCount, "images from directory");
        }
        return !images.empty();
    }

    void detectAndMatch(const cv::Mat& img1, const cv::Mat& img2,
                       std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2) {
        Logger::debug("Detecting and matching features between image pair");
        
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        cv::Mat descriptors1, descriptors2;
        
        feature_detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
        feature_detector->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);
        
        Logger::debug("Found", keypoints1.size(), "keypoints in first image and", 
                     keypoints2.size(), "in second image");
        
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
        
        Logger::debug("Found", good_matches.size(), "good matches after ratio test");
    }

    void triangulatePoints(const cv::Mat& img1, const cv::Mat& img2,
                      const cv::Mat& R, const cv::Mat& t,
                      const std::vector<cv::Point2f>& pts1,
                      const std::vector<cv::Point2f>& pts2) {
    Logger::debug("Beginning point triangulation");
    size_t initial_size = point_cloud.size();
    
    // Camera matrix (assuming default parameters)
    double focal_length = std::max(img1.cols, img1.rows);
    cv::Point2d pp(img1.cols/2, img1.rows/2);
    cv::Mat K = (cv::Mat_<double>(3,3) << 
        focal_length, 0, pp.x,
        0, focal_length, pp.y,
        0, 0, 1);
    
    // Create a temporary matrix to store the result of hconcat
    cv::Mat Rt;
    cv::hconcat(R, t, Rt);  // Concatenate R and t horizontally
    
    // Projection matrices
    cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_64F);
    cv::Mat P2 = K * Rt;  // Use the concatenated matrix
    
    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, pts1, pts2, points4D);
    
    // Convert to 3D points
    int validPoints = 0;
    for (int i = 0; i < points4D.cols; i++) {
        cv::Mat x = points4D.col(i);
        x /= x.at<float>(3);
        CloudPoint cp;
        cp.pt = cv::Point3d(x.at<float>(0), x.at<float>(1), x.at<float>(2));
        
        // Get color from first image
        cv::Point2f pt = pts1[i];
        if (pt.x >= 0 && pt.x < img1.cols && pt.y >= 0 && pt.y < img1.rows) {
            cp.color = img1.at<cv::Vec3b>(pt);
            point_cloud.push_back(cp);
            validPoints++;
        }
    }
    
    Logger::debug("Added", validPoints, "new 3D points to cloud");
}

    void savePLY(const std::string& filename) {
        Logger::info("Saving point cloud to PLY file:", filename);
        Logger::info("Total points in cloud:", point_cloud.size());
        
        try {
            std::ofstream out(filename);
            if (!out.is_open()) {
                throw std::runtime_error("Unable to open file for writing");
            }
            
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
            Logger::info("Successfully saved PLY file");
        } catch (const std::exception& e) {
            Logger::error("Failed to save PLY file:", e.what());
            throw;
        }
    }

public:
    PointCloudReconstructor() {
        feature_detector = cv::SIFT::create();
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }
    
    bool reconstruct(const std::string& input_path, const std::string& output_path) {
        Logger::info("Starting 3D reconstruction");
        try {
            if (!loadImages(input_path)) {
                Logger::error("Failed to load images");
                return false;
            }
            
            Logger::info("Processing", images.size(), "images");
            for (size_t i = 0; i < images.size() - 1; i++) {
                Logger::info("Processing image pair", i + 1, "/", images.size() - 1);
                
                std::vector<cv::Point2f> pts1, pts2;
                detectAndMatch(images[i], images[i+1], pts1, pts2);
                
                if (pts1.size() < 8) {
                    Logger::warning("Insufficient matching points (", pts1.size(), 
                                  ") for image pair", i + 1, ", skipping");
                    continue;
                }
                
                cv::Mat E = cv::findEssentialMat(pts1, pts2, 1.0, cv::Point2d(0,0));
                cv::Mat R, t;
                cv::recoverPose(E, pts1, pts2, R, t);
                
                triangulatePoints(images[i], images[i+1], R, t, pts1, pts2);
            }
            
            if (point_cloud.empty()) {
                Logger::error("Failed to reconstruct point cloud - no points generated");
                return false;
            }
            
            savePLY(output_path);
            Logger::info("Reconstruction completed successfully");
            return true;
            
        } catch (const std::exception& e) {
            Logger::error("Reconstruction failed with error:", e.what());
            throw;
        }
    }
};

int main(int argc, char** argv) {
    std::string input_path, output_path;
    bool debug_mode = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-i" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--debug") {
            debug_mode = true;
        }
    }
    
    if (input_path.empty() || output_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " -i <input_path> -o <output_path> [--debug]\n";
        return 1;
    }
    
    if (debug_mode) {
        Logger::setLevel(Logger::DEBUG);
    }
    
    try {
        PointCloudReconstructor reconstructor;
        if (!reconstructor.reconstruct(input_path, output_path)) {
            return 1;
        }
    } catch (const std::exception& e) {
        Logger::error("Fatal error:", e.what());
        return 1;
    }
    
    return 0;
}