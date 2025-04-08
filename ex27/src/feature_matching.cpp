#include "feature_matching.h"
#include <iostream>

int main() {
    std::vector<std::vector<float>> desc1 = {{1, 2, 3}, {4, 5, 6}};
    std::vector<std::vector<float>> desc2 = {{1, 2, 3}, {4, 5, 7}};
    auto matches = brute_force_match(desc1, desc2);
    for (const auto& [i, j] : matches) std::cout << i << " -> " << j << "\n";
}