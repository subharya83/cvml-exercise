#include "harris.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./harris input.png output_corners.txt\n";
        return 1;
    }
    Image img = load_image(argv[1]);
    auto corners = detect_harris_corners(img);
    std::ofstream out(argv[2]);
    for (const auto& [x, y] : corners) out << x << " " << y << "\n";
}