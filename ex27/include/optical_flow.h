#pragma once
#include "image_utils.h"
#include <Eigen/Dense>
#include <vector>

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
                    float Ix = img1.data[idx + 1] - img1.data[idx - 1]; // Central difference
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