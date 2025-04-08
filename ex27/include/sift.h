#pragma once
#include "image_utils.h"
#include "harris.h"
#include "hog.h"
#include <vector>
#include <algorithm>

struct KeyPoint {
    float x, y;          // Coordinates
    float size;          // Scale
    float angle;         // Orientation
    std::vector<float> descriptor;
};

std::vector<KeyPoint> extract_sift_features(const Image& img) {
    std::vector<KeyPoint> keypoints;

    // --- Keypoint Detection ---
    #ifdef USE_SIFT
        // Multi-scale Harris with orientation
        auto corners = detect_harris_corners_multiscale(img);
        Image Ix = convolve(img, {-1, 0, 1}, 3); // Sobel X
        Image Iy = convolve(img, {-1, 0, 1}, 3); // Sobel Y
    #else
        // Original single-scale Harris
        auto corners = detect_harris_corners(img);
    #endif

    for (const auto& [x, y] : corners) {
        KeyPoint kp;
        kp.x = x;
        kp.y = y;

        #ifdef USE_SIFT
            kp.size = 5.0f * (img.width / static_cast<float>(x + 1)); // Scale-aware
            kp.angle = get_gradient_orientation(Ix, Iy, x, y); // From hog.h
        #else
            kp.size = 5.0f;
            kp.angle = 0.0f;
        #endif

        keypoints.push_back(kp);
    }

    // Descriptor computation
    auto hog = compute_hog(img); 
    const int desc_size = 36;    // 9 bins × 4 cells

    for (auto& kp : keypoints) {
        kp.descriptor.resize(desc_size);
        int center_idx = static_cast<int>(kp.y) * img.width + static_cast<int>(kp.x);
        for (int i = 0; i < desc_size; ++i) {
            kp.descriptor[i] = hog[(center_idx + i) % hog.size()];
        }
    }


    return keypoints;
}