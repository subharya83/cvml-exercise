#pragma once
#include "image_utils.h"
#include <vector>
#include <algorithm>

std::vector<std::pair<int, int>> brute_force_match(
    const std::vector<std::vector<float>>& desc1,
    const std::vector<std::vector<float>>& desc2,
    float threshold = 0.7
) {
    std::vector<std::pair<int, int>> matches;
    for (size_t i = 0; i < desc1.size(); ++i) {
        float best_dist = std::numeric_limits<float>::max();
        int best_j = -1;
        for (size_t j = 0; j < desc2.size(); ++j) {
            float dist = 0.0f;
            for (size_t k = 0; k < desc1[i].size(); ++k) {
                dist += (desc1[i][k] - desc2[j][k]) * (desc1[i][k] - desc2[j][k]);
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_j = j;
            }
        }
        if (best_dist < threshold) matches.emplace_back(i, best_j);
    }
    return matches;
}