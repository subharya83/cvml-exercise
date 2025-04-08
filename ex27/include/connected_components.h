#pragma once
#include "image_utils.h"
#include <vector>
#include <unordered_map>

struct UnionFind {
    std::vector<int> parent;
    UnionFind(int n) : parent(n) { for (int i = 0; i < n; ++i) parent[i] = i; }
    int find(int x) { return parent[x] == x ? x : (parent[x] = find(parent[x])); }
    void merge(int x, int y) { parent[find(x)] = find(y); }
};

Image connected_components(const Image& binary_img) {
    int w = binary_img.width, h = binary_img.height;
    UnionFind uf(w * h);
    std::vector<int> labels(w * h, 0);
    // First pass: assign labels and merge
    int next_label = 1;
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            if (binary_img.data[y * w + x] == 0) continue;
            std::vector<int> neighbors;
            for (int ny = y - 1; ny <= y + 1; ++ny)
                for (int nx = x - 1; nx <= x + 1; ++nx)
                    if (binary_img.data[ny * w + nx] > 0)
                        neighbors.push_back(labels[ny * w + nx]);
            if (neighbors.empty()) labels[y * w + x] = next_label++;
            else {
                int min_label = *std::min_element(neighbors.begin(), neighbors.end());
                labels[y * w + x] = min_label;
                for (int l : neighbors) if (l != min_label) uf.merge(l, min_label);
            }
        }
    }
    // Second pass: resolve labels
    Image result{w, h, std::vector<uint8_t>(w * h, 0)};
    std::unordered_map<int, uint8_t> final_labels;
    uint8_t color = 1;
    for (int i = 0; i < w * h; ++i) {
        if (binary_img.data[i] == 0) continue;
        int root = uf.find(labels[i]);
        if (!final_labels.count(root)) final_labels[root] = color++;
        result.data[i] = final_labels[root] * 50; // Scale for visibility
    }
    return result;
}