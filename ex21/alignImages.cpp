#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

// Function to align images using SIFT-based homography
void alignImagesSIFT(const Mat& reference, const Mat& target, Mat& alignedImage, double& tx, double& ty) {
    // Detect keypoints and compute descriptors using SIFT
    Ptr<SIFT> sift = SIFT::create();
    vector<KeyPoint> keypointsRef, keypointsTarget;
    Mat descriptorsRef, descriptorsTarget;
    sift->detectAndCompute(reference, noArray(), keypointsRef, descriptorsRef);
    sift->detectAndCompute(target, noArray(), keypointsTarget, descriptorsTarget);

    // Match descriptors using FLANN
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    vector<vector<DMatch>> knnMatches;
    matcher->knnMatch(descriptorsRef, descriptorsTarget, knnMatches, 2);

    // Filter matches using Lowe's ratio test
    vector<DMatch> goodMatches;
    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i][0].distance < 0.7 * knnMatches[i][1].distance) {
            goodMatches.push_back(knnMatches[i][0]);
        }
    }

    // Extract location of good matches
    vector<Point2f> pointsRef, pointsTarget;
    for (const auto& match : goodMatches) {
        pointsRef.push_back(keypointsRef[match.queryIdx].pt);
        pointsTarget.push_back(keypointsTarget[match.trainIdx].pt);
    }

    // Compute homography using RANSAC
    Mat homography = findHomography(pointsTarget, pointsRef, RANSAC);

    // Warp the target image using the homography
    warpPerspective(target, alignedImage, homography, reference.size());

    // Extract translation parameters from homography
    tx = homography.at<double>(0, 2);
    ty = homography.at<double>(1, 2);
}

// Function to align images using phase correlation
void alignImagesPhaseCorrelation(const Mat& reference, const Mat& target, Mat& alignedImage, double& tx, double& ty) {
    // Compute phase correlation
    Mat hann;
    createHanningWindow(hann, reference.size(), CV_32F);
    Mat referenceFloat, targetFloat;
    reference.convertTo(referenceFloat, CV_32F);
    target.convertTo(targetFloat, CV_32F);
    multiply(referenceFloat, hann, referenceFloat);
    multiply(targetFloat, hann, targetFloat);

    Mat response;
    phaseCorrelate(referenceFloat, targetFloat, response);

    // Extract translation parameters
    tx = response.x;
    ty = response.y;

    // Apply translation to the target image
    Mat translationMatrix = (Mat_<double>(2, 3) << 1, 0, -tx, 0, 1, -ty;
    warpAffine(target, alignedImage, translationMatrix, reference.size());
}

int main(int argc, char** argv) {
    // Parse command-line arguments
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " -r <reference_image> -t <target_image> -o <output_image> -a <algorithm>" << endl;
        cerr << "Algorithm: 0 for SIFT, 1 for phase correlation" << endl;
        return -1;
    }

    string referencePath = argv[2];
    string targetPath = argv[3];
    string outputPath = argv[4];
    int algorithm = stoi(argv[6]);

    // Load images
    Mat reference = imread(referencePath, IMREAD_GRAYSCALE);
    Mat target = imread(targetPath, IMREAD_GRAYSCALE);

    if (reference.empty() || target.empty()) {
        cerr << "Error: Could not load images." << endl;
        return -1;
    }

    // Align images based on the selected algorithm
    Mat alignedImage;
    double tx = 0, ty = 0;

    if (algorithm == 0) {
        cout << "Using SIFT-based homography for alignment." << endl;
        alignImagesSIFT(reference, target, alignedImage, tx, ty);
    } else if (algorithm == 1) {
        cout << "Using phase correlation for alignment." << endl;
        alignImagesPhaseCorrelation(reference, target, alignedImage, tx, ty);
    } else {
        cerr << "Invalid algorithm selection." << endl;
        return -1;
    }

    // Save the aligned image
    imwrite(outputPath, alignedImage);

    // Print translation parameters
    cout << "Translation parameters (in pixels): (tx, ty) = (" << tx << ", " << ty << ")" << endl;

    return 0;
}