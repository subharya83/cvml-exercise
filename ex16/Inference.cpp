#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>

class Inference {
public:
    std::string testClassifier(const std::string& modelFile, const std::string& testVideoPath) {
        std::ifstream inFile(modelFile, std::ios::binary);
        if (!inFile.is_open()) {
            std::cerr << "Failed to open model file." << std::endl;
            return "Unknown";
        }

        // Load model and PCA (assuming they are serialized together)
        // This part requires proper serialization/deserialization logic
        // For simplicity, we assume the model and PCA are loaded correctly
        // and proceed with the rest of the logic.

        CamMotionClassifier classifier;
        std::vector<Eigen::VectorXf> features = classifier.extractFeatures(testVideoPath);
        if (!features.empty()) {
            Eigen::VectorXf meanFeature = Eigen::VectorXf::Zero(features[0].size());
            for (const auto& f : features) {
                meanFeature += f;
            }
            meanFeature /= features.size();

            // Apply PCA and predict (assuming PCA and model are loaded)
            // Eigen::VectorXf featuresPCA = pca.transform(meanFeature);
            // std::string prediction = classifier.predict(featuresPCA);
            // return prediction;

            return "PredictedClass";  // Placeholder
        }
        return "Unknown";
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " -m <model_file> -t <test_video_path>" << std::endl;
        return 1;
    }

    std::string modelFile = argv[2];
    std::string testVideoPath = argv[4];

    Inference inference;
    std::string predictedClass = inference.testClassifier(modelFile, testVideoPath);
    std::cout << "Predicted Class: " << predictedClass << std::endl;

    return 0;
}