#include "hog.h"
#include "convolution.h"  // For convolve function
#include <iostream>
#include <fstream>

std::vector<float> compute_hog(const Image& img, int cell_size, int bin_count) {
    Image Ix = convolve(img, {-1, 0, 1}, 3); // Sobel X
    Image Iy = convolve(img, {-1, 0, 1}, 3); // Sobel Y

    std::vector<float> hog_features;
    for (int cy = 0; cy < img.height; cy += cell_size) {
        for (int cx = 0; cx < img.width; cx += cell_size) {
            std::vector<float> bins(bin_count, 0.0f);
            for (int y = cy; y < cy + cell_size && y < img.height; ++y) {
                for (int x = cx; x < cx + cell_size && x < img.width; ++x) {
                    int idx = y * img.width + x;
                    float gx = Ix.data[idx], gy = Iy.data[idx];
                    float mag = std::sqrt(gx * gx + gy * gy);
                    float angle = std::atan2(gy, gx) * 180 / M_PI;
                    if (angle < 0) angle += 180;
                    int bin = static_cast<int>(angle / (180 / bin_count)) % bin_count;
                    bins[bin] += mag;
                }
            }
            hog_features.insert(hog_features.end(), bins.begin(), bins.end());
        }
    }
    return hog_features;
}

#ifdef USE_SIFT
float get_gradient_orientation(const Image& Ix, const Image& Iy, int x, int y) {
    int idx = y * Ix.width + x;
    float gx = Ix.data[idx], gy = Iy.data[idx];
    return std::atan2(gy, gx);
}
#endif

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./hog input.png\n";
        return 1;
    }
    Image img = load_image(argv[1]);
    auto hog = compute_hog(img);
    for (float val : hog) std::cout << val << " ";
    std::cout << "\n";
    return 0;
}