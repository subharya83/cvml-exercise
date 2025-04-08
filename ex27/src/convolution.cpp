#include "convolution.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./convolution input.png output.png kernel_size\n";
        return 1;
    }
    Image img = load_image(argv[1]);
    std::vector<float> gaussian_kernel = {1, 2, 1, 2, 4, 2, 1, 2, 1}; // 3x3 Gaussian
    for (auto& k : gaussian_kernel) k /= 16.0f;
    Image blurred = convolve(img, gaussian_kernel, std::stoi(argv[3]));
    save_image(blurred, argv[2]);
}