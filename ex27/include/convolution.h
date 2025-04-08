#pragma once
#include "image_utils.h"
#include <vector>
#include <algorithm>

Image convolve(const Image& img, const std::vector<float>& kernel, int ksize) {
    Image result{img.width, img.height, std::vector<uint8_t>(img.width * img.height, 0)};
    int pad = ksize / 2;
    for (int y = pad; y < img.height - pad; ++y) {
        for (int x = pad; x < img.width - pad; ++x) {
            float sum = 0.0f;
            for (int ky = 0; ky < ksize; ++ky) {
                for (int kx = 0; kx < ksize; ++kx) {
                    int idx = (y + ky - pad) * img.width + (x + kx - pad);
                    sum += img.data[idx] * kernel[ky * ksize + kx];
                }
            }
            result.data[y * img.width + x] = static_cast<uint8_t>(std::clamp(sum, 0.0f, 255.0f));
        }
    }
    return result;
}