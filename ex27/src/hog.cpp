#include "hog.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./hog input.png\n";
        return 1;
    }
    Image img = load_image(argv[1]);
    auto hog = compute_hog(img);
    for (float val : hog) std::cout << val << " ";
    std::cout << "\n";
}