#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

using namespace cv;
using namespace std;

struct Pose {
    Mat rotation;
    Mat translation;
    double delta;
};

void printTransformJSON(const Pose& pose) {
    Mat R;
    Rodrigues(pose.rotation, R);
    
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

double calculatePositionDelta(const Mat& prev, const Mat& curr) {
    if (prev.empty()) return 0.0;
    return norm(curr - prev);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <video_file>" << endl;
        return -1;
    }

    Mat cameraMatrix = (Mat_<double>(3, 3) << 
        1527.894, 0, 962.880,
        0, 1527.894, 721.309,
        0, 0, 1);
    Mat distCoeffs = Mat::zeros(4, 1, CV_64F);

    VideoCapture video(argv[1]);
    if (!video.isOpened()) {
        cerr << "Error opening video file" << endl;
        return -1;
    }

    // Debug Frame level discrepancies
    double fps = video.get(CAP_PROP_FPS);
    double frame_count = video.get(CAP_PROP_FRAME_COUNT);
    double duration = frame_count / fps;

    cout << "Video info:\n"
     << "  FPS: " << fps << "\n"
     << "  Reported frames: " << frame_count << "\n"
     << "  Calculated duration: " << duration << "s\n"
     << "  Expected frames (6fps × 7.17s): " << 6 * 7.17 << endl;

    Ptr<ORB> orb = ORB::create();
    BFMatcher matcher(NORM_HAMMING);

    Mat prevFrame, prevGray;
    vector<KeyPoint> prevKeypoints;
    Mat prevDescriptors;
    Mat prevTranslation;
    
    while (true) {
        Mat frame;
        video >> frame;
        if (frame.empty()) break;

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<KeyPoint> keypoints;
        Mat descriptors;
        orb->detectAndCompute(gray, noArray(), keypoints, descriptors);

        if (!prevFrame.empty() && !prevDescriptors.empty() && !descriptors.empty()) {
            vector<DMatch> matches;
            matcher.match(prevDescriptors, descriptors, matches);

            double minDist = DBL_MAX;
            for (const auto& m : matches) {
                if (m.distance < minDist) minDist = m.distance;
            }
            
            vector<Point2f> matchedPoints, prevMatchedPoints;
            for (const auto& m : matches) {
                if (m.distance <= max(2 * minDist, 30.0)) {
                    prevMatchedPoints.push_back(prevKeypoints[m.queryIdx].pt);
                    matchedPoints.push_back(keypoints[m.trainIdx].pt);
                }
            }

            if (prevMatchedPoints.size() > 10) {
                Mat E, mask;
                E = findEssentialMat(prevMatchedPoints, matchedPoints, cameraMatrix, RANSAC, 0.999, 1.0, mask);

                Mat R, t;
                recoverPose(E, prevMatchedPoints, matchedPoints, cameraMatrix, R, t, mask);

                Mat rvec;
                Rodrigues(R, rvec);
                
                Pose pose;
                pose.rotation = rvec;
                pose.translation = t;
                pose.delta = calculatePositionDelta(prevTranslation, t);
                printTransformJSON(pose);
                
                prevTranslation = t.clone();
            } else {
                Pose pose;
                pose.rotation = Mat::zeros(3, 1, CV_64F);
                pose.translation = Mat::zeros(3, 1, CV_64F);
                pose.delta = 0.0;
                printTransformJSON(pose);
            }
        }

        prevFrame = frame.clone();
        prevGray = gray.clone();
        prevKeypoints = keypoints;
        prevDescriptors = descriptors.clone();
    }

    return 0;
}