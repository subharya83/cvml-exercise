#include <iostream>  // Must be first for std::cerr
#include "optical_flow.h"
#include <cassert>
#include <string>

void lucas_kanade(const Image& img1, const Image& img2, int window_size) {
    assert(img1.width == img2.width && img1.height == img2.height);
    int w = img1.width, h = img1.height;
    for (int y = window_size; y < h - window_size; ++y) {
        for (int x = window_size; x < w - window_size; ++x) {
            Eigen::Matrix2f ATA = Eigen::Matrix2f::Zero();
            Eigen::Vector2f ATb = Eigen::Vector2f::Zero();
            for (int dy = -window_size; dy <= window_size; ++dy) {
                for (int dx = -window_size; dx <= window_size; ++dx) {
                    int idx = (y + dy) * w + (x + dx);
                    float Ix = img1.data[idx + 1] - img1.data[idx - 1];
                    float Iy = img1.data[idx + w] - img1.data[idx - w];
                    float It = img2.data[idx] - img1.data[idx];
                    ATA(0, 0) += Ix * Ix;
                    ATA(0, 1) += Ix * Iy;
                    ATA(1, 1) += Iy * Iy;
                    ATb(0) += -Ix * It;
                    ATb(1) += -Iy * It;
                }
            }
            ATA(1, 0) = ATA(0, 1);
            Eigen::Vector2f flow = ATA.ldlt().solve(ATb);
            std::cout << "Flow at (" << x << "," << y << "): " << flow.transpose() << "\n";
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./optical_flow img1.png img2.png window_size\n";
        return 1;
    }
    Image img1 = load_image(argv[1]);
    Image img2 = load_image(argv[2]);
    lucas_kanade(img1, img2, std::stoi(argv[3]));
    return 0;
}