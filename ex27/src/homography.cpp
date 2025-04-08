#include "homography.h"
#include <iostream>

int main() {
    std::vector<Eigen::Vector2f> src = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    std::vector<Eigen::Vector2f> dst = {{10, 10}, {20, 10}, {20, 20}, {10, 20}};
    Eigen::Matrix3f H = find_homography(src, dst);
    std::cout << "Homography matrix:\n" << H << "\n";
}