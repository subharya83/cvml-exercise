#pragma once
#include "image_utils.h"
#include <algorithm>

Image dilate(const Image& img, int kernel_size = 3) {
    Image result = img;
    int pad = kernel_size / 2;
    for (int y = pad; y < img.height - pad; ++y) {
        for (int x = pad; x < img.width - pad; ++x) {
            uint8_t max_val = 0;
            for (int ky = -pad; ky <= pad; ++ky) {
                for (int kx = -pad; kx <= pad; ++kx) {
                    int idx = (y + ky) * img.width + (x + kx);
                    max_val = std::max(max_val, img.data[idx]);
                }
            }
            result.data[y * img.width + x] = max_val;
        }
    }
    return result;
}

Image erode(const Image& img, int kernel_size = 3) {
    Image result = img;
    int pad = kernel_size / 2;
    for (int y = pad; y < img.height - pad; ++y) {
        for (int x = pad; x < img.width - pad; ++x) {
            uint8_t min_val = 255;
            for (int ky = -pad; ky <= pad; ++ky) {
                for (int kx = -pad; kx <= pad; ++kx) {
                    int idx = (y + ky) * img.width + (x + kx);
                    min_val = std::min(min_val, img.data[idx]);
                }
            }
            result.data[y * img.width + x] = min_val;
        }
    }
    return result;
}