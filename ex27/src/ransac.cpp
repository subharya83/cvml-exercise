#include "ransac.h"
#include <iostream>

int main() {
    std::vector<Point> points = {{1, 2}, {2, 4}, {3, 6}, {10, 20}, {100, 200}};
    auto [a, b] = ransac_line(points);
    std::cout << "Best fit line: y = " << a << "x + " << b << "\n";
}