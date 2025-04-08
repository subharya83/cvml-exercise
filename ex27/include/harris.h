#pragma once
#include "image_utils.h"
#include <vector>
#include <cmath>
#include <algorithm>

std::vector<std::pair<int, int>> detect_harris_corners(const Image& img, float k = 0.04, float threshold = 0.01) {
    Image Ix = convolve(img, {-1, 0, 1, -2, 0, 2, -1, 0, 1}, 3); // Sobel X
    Image Iy = convolve(img, {-1, -2, -1, 0, 0, 0, 1, 2, 1}, 3); // Sobel Y

    std::vector<std::pair<int, int>> corners;
    for (int y = 1; y < img.height - 1; ++y) {
        for (int x = 1; x < img.width - 1; ++x) {
            float Ixx = 0, Ixy = 0, Iyy = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int idx = (y + dy) * img.width + (x + dx);
                    float ix = Ix.data[idx], iy = Iy.data[idx];
                    Ixx += ix * ix;
                    Ixy += ix * iy;
                    Iyy += iy * iy;
                }
            }
            float det = Ixx * Iyy - Ixy * Ixy;
            float trace = Ixx + Iyy;
            float R = det - k * trace * trace;
            if (R > threshold) corners.emplace_back(x, y);
        }
    }
    return corners;
}