#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

class CamMotionClassifier {
public:
    CamMotionClassifier() {
        surf = cv::xfeatures2d::SURF::create();
    }

    cv::Mat computeHomography(const cv::Mat& frame1, const cv::Mat& frame2) {
        std::vector<cv::KeyPoint> kp1, kp2;
        cv::Mat des1, des2;

        surf->detectAndCompute(frame1, cv::noArray(), kp1, des1);
        surf->detectAndCompute(frame2, cv::noArray(), kp2, des2);

        cv::FlannBasedMatcher matcher(new cv::flann::KDTreeIndexParams(5), new cv::flann::SearchParams(50));
        std::vector<std::vector<cv::DMatch>> matches;
        matcher.knnMatch(des1, des2, matches, 2);

        std::vector<cv::DMatch> goodMatches;
        for (const auto& m : matches) {
            if (m[0].distance < 0.7 * m[1].distance) {
                goodMatches.push_back(m[0]);
            }
        }

        if (goodMatches.size() > 10) {
            std::vector<cv::Point2f> srcPoints, dstPoints;
            for (const auto& m : goodMatches) {
                srcPoints.push_back(kp1[m.queryIdx].pt);
                dstPoints.push_back(kp2[m.trainIdx].pt);
            }
            return cv::findHomography(srcPoints, dstPoints, cv::RANSAC, 5.0);
        }
        return cv::Mat();
    }

    Eigen::VectorXf lieAlgebraMapping(const cv::Mat& H) {
        Eigen::MatrixXf M = H;
        Eigen::MatrixXf logM = M.log();
        return Eigen::Map<Eigen::VectorXf>(logM.data(), logM.size());
    }

    std::vector<Eigen::VectorXf> extractFeatures(const std::string& videoPath, int frameSkip = 4) {
        cv::VideoCapture cap(videoPath);
        std::vector<Eigen::VectorXf> features;
        cv::Mat prevFrame;

        while (cap.isOpened()) {
            cv::Mat frame;
            if (!cap.read(frame)) break;

            if (!prevFrame.empty()) {
                cv::Mat H = computeHomography(prevFrame, frame);
                if (!H.empty()) {
                    Eigen::VectorXf M = lieAlgebraMapping(H);
                    features.push_back(M);
                }
            }

            prevFrame = frame;
            for (int i = 0; i < frameSkip; ++i) {
                cap.read(frame);  // Skip frames
            }
        }

        cap.release();
        return features;
    }

    void createDataset(const std::string& videoDir, const std::string& outputFile) {
        std::vector<Eigen::VectorXf> X;
        std::vector<std::string> y;

        for (const auto& entry : fs::directory_iterator(videoDir)) {
            if (entry.path().extension() == ".mp4") {
                std::string videoPath = entry.path().string();
                std::string shotName = entry.path().stem().string().substr(0, entry.path().stem().string().find('_'));

                std::vector<Eigen::VectorXf> features = extractFeatures(videoPath);
                if (!features.empty()) {
                    Eigen::VectorXf meanFeature = Eigen::VectorXf::Zero(features[0].size());
                    for (const auto& f : features) {
                        meanFeature += f;
                    }
                    meanFeature /= features.size();
                    X.push_back(meanFeature);
                    y.push_back(shotName);
                }
            }
        }

        std::ofstream outFile(outputFile, std::ios::binary);
        if (outFile.is_open()) {
            size_t numSamples = X.size();
            outFile.write(reinterpret_cast<char*>(&numSamples), sizeof(numSamples));
            for (size_t i = 0; i < numSamples; ++i) {
                size_t featureSize = X[i].size();
                outFile.write(reinterpret_cast<char*>(&featureSize), sizeof(featureSize));
                outFile.write(reinterpret_cast<char*>(X[i].data()), featureSize * sizeof(float));
                size_t labelSize = y[i].size();
                outFile.write(reinterpret_cast<char*>(&labelSize), sizeof(labelSize));
                outFile.write(y[i].data(), labelSize);
            }
            outFile.close();
        }
    }

private:
    cv::Ptr<cv::xfeatures2d::SURF> surf;
};