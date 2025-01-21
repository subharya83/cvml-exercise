// main.cpp
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <algorithm>
#include "audiofile.h"

struct Detection {
    double timestamp;
    double correlation;
};

std::vector<Detection> findPattern(const std::vector<double>& source, const std::vector<double>& pattern, double sampleRate, double threshold = 0.7) {
    std::vector<Detection> detections;
    int sourceSize = source.size();
    int patternSize = pattern.size();
    
    // Normalize pattern
    double patternSum = 0;
    for (double val : pattern) {
        patternSum += val * val;
    }
    
    // Sliding window cross-correlation
    for (int i = 0; i <= sourceSize - patternSize; i++) {
        double correlation = 0;
        double windowSum = 0;
        
        for (int j = 0; j < patternSize; j++) {
            correlation += source[i + j] * pattern[j];
            windowSum += source[i + j] * source[i + j];
        }
        
        // Normalize correlation
        correlation = correlation / sqrt(patternSum * windowSum);
        
        if (correlation > threshold) {
            Detection det;
            det.timestamp = static_cast<double>(i) / sampleRate;
            det.correlation = correlation;
            detections.push_back(det);
        }
    }
    
    return detections;
}

int main(int argc, char* argv[]) {
    std::string inputFile, queryFile, outputFile;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i += 2) {
        if (std::string(argv[i]) == "-i") inputFile = argv[i + 1];
        else if (std::string(argv[i]) == "-q") queryFile = argv[i + 1];
        else if (std::string(argv[i]) == "-o") outputFile = argv[i + 1];
    }
    
    if (inputFile.empty() || queryFile.empty() || outputFile.empty()) {
        std::cerr << "Usage: " << argv[0] << " -i <input_audio> -q <query_audio> -o <output_csv>" << std::endl;
        return 1;
    }
    
    AudioFile<double> inputAudio;
    AudioFile<double> queryAudio;
    
    if (!inputAudio.load(inputFile) || !queryAudio.load(queryFile)) {
        std::cerr << "Error loading audio files" << std::endl;
        return 1;
    }
    
    // Convert to mono if necessary and get samples
    std::vector<double> inputSamples = inputAudio.samples[0];
    std::vector<double> querySamples = queryAudio.samples[0];
    
    // Find matches
    std::vector<Detection> detections = findPattern(inputSamples, querySamples, inputAudio.getSampleRate());
    
    // Write results to CSV
    std::ofstream outFile(outputFile);
    outFile << "Timestamp (seconds),Correlation\n";
    for (const auto& det : detections) {
        outFile << det.timestamp << "," << det.correlation << "\n";
    }
    outFile.close();
    
    return 0;
}