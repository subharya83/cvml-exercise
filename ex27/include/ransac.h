#pragma once
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

struct Point { float x, y; };

std::pair<float, float> ransac_line(const std::vector<Point>& points, int max_iters = 1000, float threshold = 1.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, points.size() - 1);

    int best_inliers = 0;
    float best_a = 0, best_b = 0;

    for (int iter = 0; iter < max_iters; ++iter) {
        int i1 = dis(gen), i2 = dis(gen);
        if (i1 == i2) continue;

        float x1 = points[i1].x, y1 = points[i1].y;
        float x2 = points[i2].x, y2 = points[i2].y;

        float a = (y2 - y1) / (x2 - x1);
        float b = y1 - a * x1;

        int inliers = 0;
        for (const auto& p : points) {
            float dist = std::abs(a * p.x - p.y + b) / std::sqrt(a * a + 1);
            if (dist < threshold) inliers++;
        }

        if (inliers > best_inliers) {
            best_inliers = inliers;
            best_a = a;
            best_b = b;
        }
    }
    return {best_a, best_b};
}