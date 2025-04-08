#include "kmeans.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./kmeans input.png output.png k\n";
        return 1;
    }
    Image img = load_image(argv[1]);
    Image quantized = kmeans_color_quantization(img, std::stoi(argv[3]));
    save_image(quantized, argv[2]);
}