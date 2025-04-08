#pragma once
#include <Eigen/Dense>
#include <vector>

Eigen::Matrix3f find_homography(const std::vector<Eigen::Vector2f>& src,
                                const std::vector<Eigen::Vector2f>& dst) {
    assert(src.size() >= 4 && dst.size() >= 4);
    Eigen::MatrixXf A(2 * src.size(), 9);
    for (size_t i = 0; i < src.size(); ++i) {
        float x = src[i].x(), y = src[i].y();
        float u = dst[i].x(), v = dst[i].y();
        A.row(2 * i) << -x, -y, -1, 0, 0, 0, x * u, y * u, u;
        A.row(2 * i + 1) << 0, 0, 0, -x, -y, -1, x * v, y * v, v;
    }
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXf h = svd.matrixV().col(8);
    Eigen::Matrix3f H;
    H << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), h(8);
    return H / H(2, 2); // Normalize
}