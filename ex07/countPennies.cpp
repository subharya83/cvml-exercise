#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <string>

class PennyDetector {
private:
    cv::dnn::Net model;
    bool has_model;

public:
    PennyDetector() : has_model(false) {}

    bool loadModel(const std::string& model_path) {
        try {
            model = cv::dnn::readNet(model_path);
            has_model = true;
            return true;
        }
        catch (const cv::Exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            return false;
        }
    }

    int detectAndCountPennies(const std::string& image_path) {
        // Load the image
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Error: Image not found." << std::endl;
            return 0;
        }

        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // Apply Gaussian blur
        cv::Mat blurred;
        cv::GaussianBlur(gray, blurred, cv::Size(9, 9), 2);

        // Detect circles using Hough Circle Transform
        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(
            blurred,
            circles,
            cv::HOUGH_GRADIENT,
            1.2,
            30,    // minDist
            50,    // param1
            30,    // param2
            10,    // minRadius
            50     // maxRadius
        );

        if (!circles.empty()) {
            int penny_count = 0;

            for (const auto& circle : circles) {
                int x = cvRound(circle[0]);
                int y = cvRound(circle[1]);
                int radius = cvRound(circle[2]);

                // Create mask for ROI
                cv::Mat mask = cv::Mat::zeros(gray.size(), gray.type());
                cv::circle(mask, cv::Point(x, y), radius, cv::Scalar(255), -1);

                // Extract ROI
                cv::Mat coin_roi;
                image.copyTo(coin_roi, mask);

                // Measure diameter
                int diameter = radius * 2;

                // Check if the coin matches penny dimensions
                if (diameter >= 18 && diameter <= 20) {
                    penny_count++;

                    // Draw detection on image
                    cv::circle(image, cv::Point(x, y), radius, cv::Scalar(0, 255, 0), 2);
                    cv::putText(image, "Penny", 
                        cv::Point(x - radius, y - radius - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                }
            }

            // Show result
            cv::imshow("Detected Pennies", image);
            cv::waitKey(0);
            cv::destroyAllWindows();

            return penny_count;
        }

        std::cout << "No coins detected." << std::endl;
        return 0;
    }

    int detectAndCountPenniesWithCNN(const std::string& image_path) {
        if (!has_model) {
            std::cerr << "Error: CNN model not loaded" << std::endl;
            return 0;
        }

        // Load the image
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Error: Image not found." << std::endl;
            return 0;
        }

        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // Apply Gaussian blur
        cv::Mat blurred;
        cv::GaussianBlur(gray, blurred, cv::Size(9, 9), 2);

        // Detect circles using Hough Circle Transform
        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(
            blurred,
            circles,
            cv::HOUGH_GRADIENT,
            1.2,
            30,    // minDist
            50,    // param1
            30,    // param2
            10,    // minRadius
            50     // maxRadius
        );

        if (!circles.empty()) {
            int penny_count = 0;

            for (const auto& circle : circles) {
                int x = cvRound(circle[0]);
                int y = cvRound(circle[1]);
                int radius = cvRound(circle[2]);

                // Extract ROI
                cv::Rect roi(
                    std::max(0, x - radius),
                    std::max(0, y - radius),
                    std::min(image.cols - x + radius, radius * 2),
                    std::min(image.rows - y + radius, radius * 2)
                );
                cv::Mat coin_roi = image(roi);

                // Prepare ROI for CNN model
                cv::Mat coin_roi_resized;
                cv::resize(coin_roi, coin_roi_resized, cv::Size(224, 224));
                
                // Convert to blob for DNN
                cv::Mat blob = cv::dnn::blobFromImage(coin_roi_resized, 1.0, 
                    cv::Size(224, 224), cv::Scalar(0, 0, 0), true, false);

                // Run inference
                model.setInput(blob);
                cv::Mat outputs = model.forward();

                // Process detection results
                float confidence = outputs.at<float>(0);
                if (confidence > 0.5) {  // Assuming binary classification
                    penny_count++;

                    // Draw detection on image
                    cv::circle(image, cv::Point(x, y), radius, cv::Scalar(0, 255, 0), 2);
                    cv::putText(image, "Penny", 
                        cv::Point(x - radius, y - radius - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                }
            }

            // Show result
            cv::imshow("Detected Pennies", image);
            cv::waitKey(0);
            cv::destroyAllWindows();

            return penny_count;
        }

        std::cout << "No coins detected." << std::endl;
        return 0;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    PennyDetector detector;
    
    // Uncomment and modify path to use CNN-based detection
    // if (!detector.loadModel("penny_detection_model.xml")) {
    //     return 1;
    // }
    // int num_pennies = detector.detectAndCountPenniesWithCNN(argv[1]);
    
    int num_pennies = detector.detectAndCountPennies(argv[1]);
    std::cout << "Number of pennies detected: " << num_pennies << std::endl;

    return 0;
}