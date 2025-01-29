#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>

struct Circle {
    int x, y, radius;
};

std::vector<Circle> non_max_suppression(const std::vector<Circle>& circles, float overlap_threshold = 0.5f) {
    if (circles.empty()) {
        return {};
    }

    // Sort circles by radius (largest first)
    std::vector<Circle> sorted_circles = circles;
    std::sort(sorted_circles.begin(), sorted_circles.end(), [](const Circle& a, const Circle& b) {
        return a.radius > b.radius;
    });

    std::vector<Circle> suppressed;

    while (!sorted_circles.empty()) {
        // Take the largest circle
        Circle current = sorted_circles.front();
        suppressed.push_back(current);
        sorted_circles.erase(sorted_circles.begin());

        // Calculate overlap with remaining circles
        std::vector<Circle> to_keep;
        for (const auto& circle : sorted_circles) {
            // Calculate distance between centers
            float dx = current.x - circle.x;
            float dy = current.y - circle.y;
            float distance = std::sqrt(dx * dx + dy * dy);

            // Calculate overlap ratio (intersection over union)
            if (distance < (current.radius + circle.radius)) {
                float overlap_ratio = (current.radius * current.radius) / (circle.radius * circle.radius);
                if (overlap_ratio <= overlap_threshold) {
                    to_keep.push_back(circle);
                }
            } else {
                to_keep.push_back(circle);
            }
        }

        sorted_circles = to_keep;
    }

    return suppressed;
}

int detect_circles(const std::string& image_path, bool visualize = false) {
    // Load the image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Image not found." << std::endl;
        return 0;
    }

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Apply Gaussian blur to reduce noise
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(9, 9), 2);

    // Apply Canny edge detection
    cv::Mat edges;
    cv::Canny(blurred, edges, 50, 150);

    // Create a binary version of the image
    cv::Mat binary;
    cv::threshold(edges, binary, 50, 255, cv::THRESH_BINARY);

    // Detect circles using Hough Circle Transform on the binarized image
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(binary, circles, cv::HOUGH_GRADIENT, 1.2, 30, 50, 30, 10, 100);

    if (!circles.empty()) {
        // Convert to Circle struct
        std::vector<Circle> detected_circles;
        for (const auto& circle : circles) {
            detected_circles.push_back({static_cast<int>(circle[0]), static_cast<int>(circle[1]), static_cast<int>(circle[2])});
        }

        // Apply Non-Maximum Suppression (NMS)
        std::vector<Circle> filtered_circles = non_max_suppression(detected_circles, 0.5f);

        if (!filtered_circles.empty()) {
            int num_circles = filtered_circles.size();

            if (visualize) {
                // Draw the detected circles on the original image
                for (const auto& circle : filtered_circles) {
                    // Draw the outer circle
                    cv::circle(image, cv::Point(circle.x, circle.y), circle.radius, cv::Scalar(0, 255, 0), 2);
                    // Draw the center of the circle
                    cv::circle(image, cv::Point(circle.x, circle.y), 2, cv::Scalar(0, 0, 255), 3);
                }

                // Show the output image with detected circles
                cv::imshow("Detected Circles", image);
                cv::waitKey(0);
                cv::destroyAllWindows();
            }

            return num_circles;
        } else {
            std::cout << "No circular objects detected after NMS." << std::endl;
            return 0;
        }
    } else {
        std::cout << "No circular objects detected." << std::endl;
        return 0;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " -i <image_path> [-v]" << std::endl;
        return -1;
    }

    std::string image_path;
    bool visualize = false;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-i" && i + 1 < argc) {
            image_path = argv[i + 1];
        } else if (std::string(argv[i]) == "-v") {
            visualize = true;
        }
    }

    if (image_path.empty()) {
        std::cerr << "Error: Image path not provided." << std::endl;
        return -1;
    }

    // Detect circles in the image
    int num_circles = detect_circles(image_path, visualize);

    // Print the number of detected circles
    std::cout << "Number of circular objects detected: " << num_circles << std::endl;

    return 0;
}