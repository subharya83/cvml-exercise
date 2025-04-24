#include "harris.h"
#include "convolution.h"
#include <iostream>
#include <fstream>

std::vector<std::pair<int, int>> detect_harris_corners(const Image& img, float k, float threshold) {
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

#ifdef USE_SIFT
std::vector<std::pair<int, int>> detect_harris_corners_multiscale(const Image& img, 
                                                                const std::vector<float>& scales) {
    std::vector<std::pair<int, int>> all_corners;
    for (float scale : scales) {
        int new_w = static_cast<int>(img.width * scale);
        int new_h = static_cast<int>(img.height * scale);
        Image resized{new_w, new_h, std::vector<uint8_t>(new_w * new_h)};
        
        for (int y = 0; y < new_h; ++y) {
            for (int x = 0; x < new_w; ++x) {
                int orig_x = static_cast<int>(x / scale);
                int orig_y = static_cast<int>(y / scale);
                resized.data[y * new_w + x] = img.data[orig_y * img.width + orig_x];
            }
        }
        
        auto corners = detect_harris_corners(resized);
        for (auto& [x, y] : corners) {
            all_corners.emplace_back(static_cast<int>(x / scale), 
                                   static_cast<int>(y / scale));
        }
    }
    return all_corners;
}
#endif

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./harris input.png output_corners.txt\n";
        return 1;
    }
    
    Image img = load_image(argv[1]);
    auto corners = detect_harris_corners(img);
    
    std::ofstream out(argv[2]);
    for (const auto& [x, y] : corners) {
        out << x << " " << y << "\n";
    }
    
    return 0;
}