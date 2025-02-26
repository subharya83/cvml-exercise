#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

class DimensionEstimator {
public:
    DimensionEstimator() {
        // Known dimensions of a standard soda can in cm
        soda_can_height = 12.2;  // standard height in cm
        soda_can_diameter = 6.6; // standard diameter in cm

        // Load YOLOv5 model
        loadYOLOModel("weights/yolov5s.onnx");
    }

    void estimateDimensions(const std::string& img_path, const std::string& output_path, const std::string& target_type) {
        // Load image
        cv::Mat img = cv::imread(img_path);
        if (img.empty()) {
            std::cerr << "Error: Could not load image!" << std::endl;
            return;
        }

        // Preprocess image for YOLO
        cv::Mat blob;
        preprocessImage(img, blob);

        // Run YOLO inference
        std::vector<cv::Mat> outputs;
        runInference(blob, outputs);

        // Process detections
        std::vector<cv::Rect> detections = processDetections(outputs, img.size());

        // Find soda can and target object
        cv::Rect soda_can_bbox, target_bbox;
        findObjects(detections, soda_can_bbox, target_bbox, target_type);

        // Estimate dimensions
        if (!soda_can_bbox.empty() && !target_bbox.empty()) {
            float scale_factor = soda_can_height / (soda_can_bbox.height);
            float estimated_height_m = (target_bbox.height * scale_factor) / 100;
            float estimated_width_m = (target_bbox.width * scale_factor) / 100;

            // Draw results
            drawResults(img, soda_can_bbox, target_bbox, estimated_height_m, estimated_width_m, output_path);

            std::cout << "Estimated dimensions:" << std::endl;
            std::cout << "Height: " << estimated_height_m << " meters" << std::endl;
            std::cout << "Width: " << estimated_width_m << " meters" << std::endl;
            std::cout << "Reference object: soda can (height: " << soda_can_height << " cm)" << std::endl;
        } else {
            std::cerr << "Error: Could not detect soda can or target object in the image." << std::endl;
        }
    }

private:
    float soda_can_height;
    float soda_can_diameter;
    cv::dnn::Net yolo_net;

    void loadYOLOModel(const std::string& model_path) {
        yolo_net = cv::dnn::readNetFromONNX(model_path);
        if (yolo_net.empty()) {
            std::cerr << "Error: Could not load YOLO model!" << std::endl;
            exit(1);
        }
    }

    void preprocessImage(cv::Mat& img, cv::Mat& blob) {
        cv::dnn::blobFromImage(img, blob, 1 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    }

    void runInference(cv::Mat& blob, std::vector<cv::Mat>& outputs) {
        yolo_net.setInput(blob);
        yolo_net.forward(outputs, yolo_net.getUnconnectedOutLayersNames());
    }

    std::vector<cv::Rect> processDetections(const std::vector<cv::Mat>& outputs, const cv::Size& img_size) {
        std::vector<cv::Rect> detections;
        for (const auto& output : outputs) {
            for (int i = 0; i < output.rows; i++) {
                float confidence = output.at<float>(i, 4);
                if (confidence > 0.5) {
                    int class_id = static_cast<int>(output.at<float>(i, 5));
                    int x = static_cast<int>(output.at<float>(i, 0) * img_size.width);
                    int y = static_cast<int>(output.at<float>(i, 1) * img_size.height);
                    int w = static_cast<int>(output.at<float>(i, 2) * img_size.width);
                    int h = static_cast<int>(output.at<float>(i, 3) * img_size.height);
                    detections.emplace_back(x, y, w, h);
                }
            }
        }
        return detections;
    }

    void findObjects(const std::vector<cv::Rect>& detections, cv::Rect& soda_can_bbox, cv::Rect& target_bbox, const std::string& target_type) {
        for (const auto& bbox : detections) {
            // Assuming class_id 39 is soda can (bottle)
            if (true) {  // Replace with actual class_id check
                soda_can_bbox = bbox;
            }
            // Assuming class_id 58 is potted plant (tree)
            if (target_type == "tree" && true) {  // Replace with actual class_id check
                target_bbox = bbox;
            }
        }
    }

    void drawResults(cv::Mat& img, const cv::Rect& soda_can_bbox, const cv::Rect& target_bbox, float height_m, float width_m, const std::string& output_path) {
        cv::rectangle(img, soda_can_bbox, cv::Scalar(255, 0, 0), 2);
        cv::putText(img, "Soda Can (Reference)", cv::Point(soda_can_bbox.x, soda_can_bbox.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 0, 0), 2);

        cv::rectangle(img, target_bbox, cv::Scalar(0, 0, 255), 2);
        cv::putText(img, "Height: " + std::to_string(height_m) + "m, Width: " + std::to_string(width_m) + "m", cv::Point(target_bbox.x, target_bbox.y - 30), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);

        cv::imwrite(output_path, img);
        std::cout << "Result saved as " << output_path << std::endl;
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image> [tree|building]" << std::endl;
        return 1;
    }

    std::string img_path = argv[1];
    std::string output_path = argv[2];
    std::string target_type = (argc > 3) ? argv[3] : "tree";

    DimensionEstimator estimator;
    estimator.estimateDimensions(img_path, output_path, target_type);

    return 0;
}