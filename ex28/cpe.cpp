#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

using namespace cv;
using namespace std;

// Camera intrinsic parameters
const Mat cameraMatrix = (Mat_<double>(3, 3) << 
    1527.894, 0, 962.880,
    0, 1527.894, 721.309,
    0, 0, 1);

// Assuming no lens distortion for simplicity
const Mat distCoeffs = Mat::zeros(4, 1, CV_64F);

struct Pose {
    Mat rotation;    // Rotation vector
    Mat translation; // Translation vector
    double delta;    // Position delta from previous frame
};

void printTransformJSON(const Pose& pose) {
    Mat R;
    Rodrigues(pose.rotation, R); // Convert rotation vector to matrix
    
    // Create 4x4 transformation matrix
    Mat transform = Mat::eye(4, 4, CV_64F);
    R.copyTo(transform(Rect(0, 0, 3, 3)));
    pose.translation.copyTo(transform(Rect(3, 0, 1, 3)));
    
    cout << fixed << setprecision(15);
    cout << "{\"transform\": [";
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            cout << transform.at<double>(i, j);
            if (i != 3 || j != 3) cout << ", ";
        }
    }
    cout << "], \"position_delta\": " << pose.delta << "}" << endl;
}

double calculatePositionDelta(const Mat& prevTranslation, const Mat& currTranslation) {
    if (prevTranslation.empty()) return 0.0;
    return norm(currTranslation - prevTranslation);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <video_file>" << endl;
        return -1;
    }

    // Open video file
    VideoCapture video(argv[1]);
    if (!video.isOpened()) {
        cerr << "Error: Could not open video file." << endl;
        return -1;
    }

    // Feature detector and matcher
    Ptr<ORB> orb = ORB::create();
    BFMatcher matcher(NORM_HAMMING);

    Mat prevFrame, prevGray;
    vector<KeyPoint> prevKeypoints;
    Mat prevDescriptors;
    vector<Point3f> objectPoints;
    vector<Point2f> prevMatchedPoints;
    Mat prevTranslation;
    
    while (true) {
        Mat frame;
        video >> frame;
        if (frame.empty()) break;

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Detect features
        vector<KeyPoint> keypoints;
        Mat descriptors;
        orb->detectAndCompute(gray, noArray(), keypoints, descriptors);

        if (!prevFrame.empty()) {
            // Match features with previous frame
            vector<DMatch> matches;
            if (!prevDescriptors.empty() && !descriptors.empty()) {
                matcher.match(prevDescriptors, descriptors, matches);
            }

            // Filter good matches
            double minDist = DBL_MAX;
            for (const auto& m : matches) {
                if (m.distance < minDist) minDist = m.distance;
            }
            
            vector<DMatch> goodMatches;
            vector<Point2f> matchedPoints, prevMatchedPoints;
            for (const auto& m : matches) {
                if (m.distance <= max(2 * minDist, 30.0)) {
                    goodMatches.push_back(m);
                    prevMatchedPoints.push_back(prevKeypoints[m.queryIdx].pt);
                    matchedPoints.push_back(keypoints[m.trainIdx].pt);
                }
            }

            if (goodMatches.size() > 10) {
                // Find essential matrix
                Mat E, mask;
                E = findEssentialMat(prevMatchedPoints, matchedPoints, cameraMatrix, RANSAC, 0.999, 1.0, mask);

                // Recover pose
                Mat R, t;
                recoverPose(E, prevMatchedPoints, matchedPoints, cameraMatrix, R, t, mask);

                // Convert rotation matrix to rotation vector
                Mat rvec;
                Rodrigues(R, rvec);

                // Calculate position delta
                double delta = calculatePositionDelta(prevTranslation, t);
                
                // Store current pose
                Pose pose;
                pose.rotation = rvec;
                pose.translation = t;
                pose.delta = delta;
                
                // Print in JSON format
                printTransformJSON(pose);
                
                // Update previous translation
                prevTranslation = t.clone();
            } else {
                // Not enough matches - output identity transform
                Pose pose;
                pose.rotation = Mat::zeros(3, 1, CV_64F);
                pose.translation = Mat::zeros(3, 1, CV_64F);
                pose.delta = 0.0;
                printTransformJSON(pose);
            }
        }

        // Update previous frame data
        prevFrame = frame.clone();
        prevGray = gray.clone();
        prevKeypoints = keypoints;
        prevDescriptors = descriptors.clone();
    }

    return 0;
}