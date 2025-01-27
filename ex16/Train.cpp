#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <filesystem>

namespace fs = std::filesystem;

class Train {
public:
    void loadDataset(const std::string& datasetFile, std::vector<Eigen::VectorXf>& X, std::vector<std::string>& y) {
        std::ifstream inFile(datasetFile, std::ios::binary);
        if (inFile.is_open()) {
            size_t numSamples;
            inFile.read(reinterpret_cast<char*>(&numSamples), sizeof(numSamples));
            X.resize(numSamples);
            y.resize(numSamples);
            for (size_t i = 0; i < numSamples; ++i) {
                size_t featureSize;
                inFile.read(reinterpret_cast<char*>(&featureSize), sizeof(featureSize));
                X[i].resize(featureSize);
                inFile.read(reinterpret_cast<char*>(X[i].data()), featureSize * sizeof(float));
                size_t labelSize;
                inFile.read(reinterpret_cast<char*>(&labelSize), sizeof(labelSize));
                y[i].resize(labelSize);
                inFile.read(&y[i][0], labelSize);
            }
            inFile.close();
        }
    }

    void trainModel(const std::string& videoDir, const std::string& datasetFile, const std::string& modelFile) {
        if (!fs::exists(datasetFile)) {
            std::cout << "Dataset file " << datasetFile << " not found. Creating dataset..." << std::endl;
            CamMotionClassifier classifier;
            classifier.createDataset(videoDir, datasetFile);
        }

        std::vector<Eigen::VectorXf> X;
        std::vector<std::string> y;
        loadDataset(datasetFile, X, y);

        // Apply PCA and train SVM (assuming PCA and SVM are implemented)
        // For simplicity, we assume the PCA and SVM are trained correctly
        // and proceed with saving the model.

        std::ofstream outFile(modelFile, std::ios::binary);
        if (outFile.is_open()) {
            // Serialize PCA and SVM models
            // boost::archive::text_oarchive oa(outFile);
            // oa << pca << svm;
            outFile.close();
        }
    }
};

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " -v <video_dir> -d <dataset_file> -m <model_file>" << std::endl;
        return 1;
    }

    std::string videoDir = argv[2];
    std::string datasetFile = argv[4];
    std::string modelFile = argv[6];

    Train train;
    train.trainModel(videoDir, datasetFile, modelFile);

    return 0;
}