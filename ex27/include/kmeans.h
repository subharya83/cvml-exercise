#pragma once
#include "image_utils.h"
#include <vector>
#include <random>
#include <limits>

Image kmeans_color_quantization(const Image& img, int k, int max_iters = 100) {
    std::vector<uint8_t> centers(k);
    std::vector<int> assignments(img.data.size());

    // Initialize centers randomly
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for (int i = 0; i < k; ++i) centers[i] = dis(gen);

    for (int iter = 0; iter < max_iters; ++iter) {
        // Assign each pixel to the nearest center
        for (size_t i = 0; i < img.data.size(); ++i) {
            int best_cluster = 0;
            int min_dist = std::numeric_limits<int>::max();
            for (int j = 0; j < k; ++j) {
                int dist = abs(img.data[i] - centers[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            assignments[i] = best_cluster;
        }

        // Update centers
        std::vector<int> cluster_sums(k, 0);
        std::vector<int> cluster_counts(k, 0);
        for (size_t i = 0; i < img.data.size(); ++i) {
            int cluster = assignments[i];
            cluster_sums[cluster] += img.data[i];
            cluster_counts[cluster]++;
        }
        for (int j = 0; j < k; ++j) {
            if (cluster_counts[j] > 0) {
                centers[j] = static_cast<uint8_t>(cluster_sums[j] / cluster_counts[j]);
            }
        }
    }

    // Generate quantized image
    Image result = img;
    for (size_t i = 0; i < result.data.size(); ++i) {
        result.data[i] = centers[assignments[i]];
    }
    return result;
}