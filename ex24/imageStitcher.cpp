#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <random>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>

namespace fs = std::filesystem;

// Compute edge density using Canny edge detector
double computeEdgeDensity(const cv::Mat& image) {
    cv::Mat gray, edges;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 100, 200);
    
    int edgeCount = cv::countNonZero(edges);
    double totalPixels = static_cast<double>(image.rows * image.cols);
    return edgeCount / totalPixels;
}

// Compute color histogram for an image
std::vector<float> computeColorHistogram(const cv::Mat& image) {
    std::vector<float> hist;
    std::vector<cv::Mat> bgr_planes;
    cv::split(image, bgr_planes);
    
    int histSize = 8;
    float range[] = {0, 256};
    const float* histRange = {range};
    
    for (const auto& plane : bgr_planes) {
        cv::Mat hist_channel;
        cv::calcHist(&plane, 1, 0, cv::Mat(), hist_channel, 1, &histSize, &histRange);
        hist.insert(hist.end(), hist_channel.begin<float>(), hist_channel.end<float>());
    }
    
    return hist;
}

// Validate if an image belongs to the set based on color histogram and edge density
bool validateImage(const cv::Mat& image, 
                  const std::vector<std::vector<float>>& referenceHistograms,
                  const std::vector<double>& referenceEdgeDensities,
                  double histThreshold = 0.5,
                  double edgeThreshold = 0.5) {
    
    auto hist = computeColorHistogram(image);
    double edgeDensity = computeEdgeDensity(image);
    
    for (size_t i = 0; i < referenceHistograms.size(); i++) {
        double histSimilarity = cv::compareHist(cv::Mat(hist), cv::Mat(referenceHistograms[i]), cv::HISTCMP_CORREL);
        double edgeDiff = std::abs(edgeDensity - referenceEdgeDensities[i]) / std::max(edgeDensity, referenceEdgeDensities[i]);
        double edgeSimilarity = 1.0 - edgeDiff;
        
        if (histSimilarity > histThreshold && edgeSimilarity > edgeThreshold) {
            return true;
        }
    }
    
    return false;
}

// Calculate reference features from a subset of images
std::pair<std::vector<std::vector<float>>, std::vector<double>> 
calculateReferenceFeatures(const std::vector<cv::Mat>& images, int sampleSize = 5) {
    std::vector<std::vector<float>> referenceHistograms;
    std::vector<double> referenceEdgeDensities;
    
    std::vector<int> sampleIndices;
    if (images.size() <= static_cast<size_t>(sampleSize)) {
        for (size_t i = 0; i < images.size(); i++) {
            sampleIndices.push_back(i);
        }
    } else {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(0, images.size() - 1);
        
        while (sampleIndices.size() < static_cast<size_t>(sampleSize)) {
            int idx = dist(gen);
            if (std::find(sampleIndices.begin(), sampleIndices.end(), idx) == sampleIndices.end()) {
                sampleIndices.push_back(idx);
            }
        }
    }
    
    for (int idx : sampleIndices) {
        referenceHistograms.push_back(computeColorHistogram(images[idx]));
        referenceEdgeDensities.push_back(computeEdgeDensity(images[idx]));
    }
    
    return {referenceHistograms, referenceEdgeDensities};
}

// Stitch multiple images into a panorama
cv::Mat stitchImages(const std::vector<cv::Mat>& images) {
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create();
    cv::Mat result;
    cv::Stitcher::Status status = stitcher->stitch(images, result);
    
    if (status != cv::Stitcher::OK) {
        throw std::runtime_error("Stitching failed with error code: " + std::to_string(status));
    }
    
    return result;
}

int main(int argc, char** argv) {
    cv::CommandLineParser parser(argc, argv,
        "{help h||show help message}"
        "{i|input|input directory containing images}"
        "{o|output|output file for the stitched panorama}"
        "{hist-threshold|0.6|threshold for histogram similarity (0-1)}"
        "{edge-threshold|0.5|threshold for edge density similarity (0-1)}"
    );
    
    if (parser.has("help") || !parser.has("i") || !parser.has("o")) {
        std::cout << "Usage: " << argv[0] << " -i=<input_dir> -o=<output_file> [--hist-threshold=<val>] [--edge-threshold=<val>]" << std::endl;
        parser.printMessage();
        return 0;
    }
    
    std::string inputDir = parser.get<std::string>("i");
    std::string outputFile = parser.get<std::string>("o");
    double histThreshold = parser.get<double>("hist-threshold");
    double edgeThreshold = parser.get<double>("edge-threshold");
    
    if (!fs::exists(inputDir) || !fs::is_directory(inputDir)) {
        std::cerr << "Error: Input directory '" << inputDir << "' does not exist." << std::endl;
        return 1;
    }
    
    std::vector<cv::Mat> images;
    std::vector<std::string> validExtensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"};
    
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            
            if (std::find(validExtensions.begin(), validExtensions.end(), ext) != validExtensions.end()) {
                cv::Mat img = cv::imread(entry.path().string());
                if (!img.empty()) {
                    images.push_back(img);
                } else {
                    std::cerr << "Warning: Could not load image '" << entry.path().filename().string() << "'." << std::endl;
                }
            }
        }
    }
    
    if (images.size() < 2) {
        std::cerr << "Error: Need at least 2 valid images for stitching." << std::endl;
        return 1;
    }
    
    std::cout << "Found " << images.size() << " images." << std::endl;
    
    std::cout << "Calculating reference features..." << std::endl;
    auto [referenceHistograms, referenceEdgeDensities] = calculateReferenceFeatures(images);
    
    std::vector<cv::Mat> validImages;
    for (size_t i = 0; i < images.size(); i++) {
        if (i < referenceHistograms.size()) {
            validImages.push_back(images[i]);
            continue;
        }
        
        if (validateImage(images[i], referenceHistograms, referenceEdgeDensities, 
                         histThreshold, edgeThreshold)) {
            validImages.push_back(images[i]);
            std::cout << "Image " << i+1 << "/" << images.size() << ": Valid" << std::endl;
        } else {
            std::cout << "Image " << i+1 << "/" << images.size() << ": Invalid (filtered out)" << std::endl;
        }
    }
    
    std::cout << validImages.size() << " out of " << images.size() << " images passed validation." << std::endl;
    
    if (validImages.size() < 2) {
        std::cerr << "Error: Not enough valid images for stitching after filtering." << std::endl;
        return 1;
    }
    
    std::cout << "Stitching images..." << std::endl;
    try {
        cv::Mat result = stitchImages(validImages);
        cv::imwrite(outputFile, result);
        std::cout << "Stitched image saved to '" << outputFile << "'." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Stitching failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}