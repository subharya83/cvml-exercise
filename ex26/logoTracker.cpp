#include <opencv2/opencv.hpp>
#include <iostream>
#include "sort.h"  // Include the SORT tracker implementation

class ORBLogoDetector {
public:
    ORBLogoDetector(const cv::Mat& logo_image) : logo_image(logo_image) {
        orb = cv::ORB::create();
        bf = cv::BFMatcher(cv::NORM_HAMMING, true);
        orb->detectAndCompute(logo_image, cv::noArray(), logo_kp, logo_des);
    }

    std::vector<cv::Rect> detect(const cv::Mat& frame) {
        std::vector<cv::KeyPoint> frame_kp;
        cv::Mat frame_des;
        orb->detectAndCompute(frame, cv::noArray(), frame_kp, frame_des);

        if (!frame_des.empty() && !logo_des.empty()) {
            std::vector<cv::DMatch> matches;
            bf.match(logo_des, frame_des, matches);

            // Sort matches by distance (best matches first)
            std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
                return a.distance < b.distance;
            });

            std::vector<cv::Point2f> src_pts, dst_pts;
            for (const auto& m : matches) {
                src_pts.push_back(logo_kp[m.queryIdx].pt);
                dst_pts.push_back(frame_kp[m.trainIdx].pt);
            }

            // Compute homography
            cv::Mat M = cv::findHomography(src_pts, dst_pts, cv::RANSAC, 5.0);
            if (!M.empty()) {
                std::vector<cv::Point2f> logo_corners(4);
                logo_corners[0] = cv::Point2f(0, 0);
                logo_corners[1] = cv::Point2f(0, logo_image.rows - 1);
                logo_corners[2] = cv::Point2f(logo_image.cols - 1, logo_image.rows - 1);
                logo_corners[3] = cv::Point2f(logo_image.cols - 1, 0);
                std::vector<cv::Point2f> transformed_corners(4);
                cv::perspectiveTransform(logo_corners, transformed_corners, M);

                // Calculate bounding box
                float x1 = transformed_corners[0].x;
                float y1 = transformed_corners[0].y;
                float x2 = transformed_corners[2].x;
                float y2 = transformed_corners[2].y;

                return { cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)) };
            }
        }
        return {};  // No detection
    }

private:
    cv::Mat logo_image;
    cv::Ptr<cv::ORB> orb;
    cv::BFMatcher bf;
    std::vector<cv::KeyPoint> logo_kp;
    cv::Mat logo_des;
};

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <video_path> <logo_image_path> <output_video_path>" << std::endl;
        return -1;
    }

    // Load the logo image
    cv::Mat logo_image = cv::imread(argv[2], cv::IMREAD_COLOR);
    if (logo_image.empty()) {
        std::cerr << "Error: Could not load logo image." << std::endl;
        return -1;
    }

    // Initialize ORB detector
    ORBLogoDetector detector(logo_image);

    // Initialize SORT tracker
    Sort tracker;

    // Open the video
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video." << std::endl;
        return -1;
    }

    // Prepare video writer
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter out(argv[3], cv::VideoWriter::fourcc('M', 'P', '4', 'V'), 20.0, cv::Size(frame_width, frame_height));

    cv::Mat frame;
    while (cap.read(frame)) {
        // Detect the logo
        auto detections = detector.detect(frame);

        // Convert detections to SORT format (x1, y1, x2, y2, confidence)
        std::vector<std::vector<float>> sort_detections;
        for (const auto& rect : detections) {
            sort_detections.push_back({ static_cast<float>(rect.x), static_cast<float>(rect.y),
                                       static_cast<float>(rect.x + rect.width), static_cast<float>(rect.y + rect.height), 1.0f });
        }

        // Update SORT tracker
        auto tracks = tracker.update(sort_detections);

        // Draw tracked bounding boxes
        for (const auto& track : tracks) {
            int x1 = static_cast<int>(track[0]);
            int y1 = static_cast<int>(track[1]);
            int x2 = static_cast<int>(track[2]);
            int y2 = static_cast<int>(track[3]);
            int id = static_cast<int>(track[4]);

            cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, "ID " + std::to_string(id), cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
        }

        // Write the frame to the output video
        out.write(frame);
    }

    // Release resources
    cap.release();
    out.release();

    return 0;
}