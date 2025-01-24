#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

// Function to compute the integral image
Mat computeIntegralImage(const Mat& image) {
    Mat integral;
    integralImage(image, integral, CV_32S);
    return integral;
}

// Function to apply adaptive thresholding using integral images
Mat adaptiveThresholding(const Mat& image, int k, int C) {
    int rows = image.rows;
    int cols = image.cols;
    int halfK = k / 2;

    // Compute the integral image
    Mat integral = computeIntegralImage(image);

    // Output binary image
    Mat binaryImage = Mat::zeros(image.size(), CV_8U);

    // Iterate through each pixel
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Define the region boundaries
            int y1 = max(0, i - halfK);
            int y2 = min(rows - 1, i + halfK);
            int x1 = max(0, j - halfK);
            int x2 = min(cols - 1, j + halfK);

            // Compute the sum of pixel values in the region using the integral image
            int regionSum = integral.at<int>(y2, x2);
            if (y1 > 0) regionSum -= integral.at<int>(y1 - 1, x2);
            if (x1 > 0) regionSum -= integral.at<int>(y2, x1 - 1);
            if (y1 > 0 && x1 > 0) regionSum += integral.at<int>(y1 - 1, x1 - 1);

            // Compute the mean
            int regionSize = (y2 - y1 + 1) * (x2 - x1 + 1);
            int regionMean = regionSum / regionSize;

            // Apply the threshold
            if (image.at<uchar>(i, j) < regionMean - C) {
                binaryImage.at<uchar>(i, j) = 0;  // Black
            } else {
                binaryImage.at<uchar>(i, j) = 255;  // White
            }
        }
    }

    return binaryImage;
}

// Function to detect bright regions using integral images
Mat detectBrightRegions(const Mat& image, int w, int h, float T, vector<Point>& brightRegions) {
    int rows = image.rows;
    int cols = image.cols;
    int halfW = w / 2;
    int halfH = h / 2;

    // Compute the integral image
    Mat integral = computeIntegralImage(image);

    // Output image to highlight bright regions
    Mat outputImage = Mat::zeros(image.size(), CV_8U);

    // Iterate through each pixel
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Define the region boundaries
            int y1 = max(0, i - halfH);
            int y2 = min(rows - 1, i + halfH);
            int x1 = max(0, j - halfW);
            int x2 = min(cols - 1, j + halfW);

            // Compute the sum of pixel values in the region using the integral image
            int regionSum = integral.at<int>(y2, x2);
            if (y1 > 0) regionSum -= integral.at<int>(y1 - 1, x2);
            if (x1 > 0) regionSum -= integral.at<int>(y2, x1 - 1);
            if (y1 > 0 && x1 > 0) regionSum += integral.at<int>(y1 - 1, x1 - 1);

            // Compute the mean intensity of the region
            int regionSize = (y2 - y1 + 1) * (x2 - x1 + 1);
            float regionMean = static_cast<float>(regionSum) / regionSize;

            // Highlight the region if the mean intensity exceeds the threshold
            if (regionMean > T) {
                outputImage.at<uchar>(i, j) = 255;  // Highlight
                brightRegions.push_back(Point(j, i));  // Store the coordinates of the bright region
            }
        }
    }

    return outputImage;
}

int main(int argc, char** argv) {
    // Parse command-line arguments
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " -i <input_image> -m <mode> -o <output_image>" << endl;
        return -1;
    }

    string inputPath, outputPath;
    int mode = 1;

    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == "-i") {
            inputPath = argv[++i];
        } else if (string(argv[i]) == "-m") {
            mode = stoi(argv[++i]);
        } else if (string(argv[i]) == "-o") {
            outputPath = argv[++i];
        }
    }

    // Load the grayscale image
    Mat image = imread(inputPath, IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Error: Image not found." << endl;
        return -1;
    }

    Mat outputImage;
    vector<Point> brightRegions;  // To store coordinates of bright regions

    if (mode == 0) {
        // Parameters for adaptive thresholding
        int k = 15;  // Size of the region
        int C = 10;  // Constant to subtract from the mean

        // Apply adaptive thresholding
        outputImage = adaptiveThresholding(image, k, C);
    } else if (mode == 1) {
        // Parameters for detecting bright regions
        int w = 50;  // Width of the region
        int h = 50;  // Height of the region
        float T = 200.0f;  // Intensity threshold

        // Detect bright regions
        outputImage = detectBrightRegions(image, w, h, T, brightRegions);

        // Output the number of detected bright regions and their coordinates
        cout << "Number of bright regions detected: " << brightRegions.size() << endl;
        cout << "Coordinates of bright regions (x, y):" << endl;
        for (const auto& point : brightRegions) {
            cout << "(" << point.x << ", " << point.y << ")" << endl;
        }
    } else {
        cerr << "Invalid mode selected. Use 0 for adaptive thresholding or 1 for bright region detection." << endl;
        return -1;
    }

    // Save the output image
    imwrite(outputPath, outputImage);

    // Display the results
    imshow("Original Image", image);
    imshow("Output Image", outputImage);
    waitKey(0);
    destroyAllWindows();

    return 0;
}