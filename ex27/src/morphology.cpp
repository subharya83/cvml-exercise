#include "morphology.h"
#include <iostream>  // For std::cerr
#include <stdexcept> // For std::runtime_error
#include <string>    // For std::string

Image dilate(const Image& img, int kernel_size) {
    Image result = img;
    int pad = kernel_size / 2;
    for (int y = pad; y < img.height - pad; ++y) {
        for (int x = pad; x < img.width - pad; ++x) {
            uint8_t max_val = 0;
            for (int ky = -pad; ky <= pad; ++ky) {
                for (int kx = -pad; kx <= pad; ++kx) {
                    int idx = (y + ky) * img.width + (x + kx);
                    max_val = std::max(max_val, img.data[idx]);
                }
            }
            result.data[y * img.width + x] = max_val;
        }
    }
    return result;
}

Image erode(const Image& img, int kernel_size) {
    Image result = img;
    int pad = kernel_size / 2;
    for (int y = pad; y < img.height - pad; ++y) {
        for (int x = pad; x < img.width - pad; ++x) {
            uint8_t min_val = 255;
            for (int ky = -pad; ky <= pad; ++ky) {
                for (int kx = -pad; kx <= pad; ++kx) {
                    int idx = (y + ky) * img.width + (x + kx);
                    min_val = std::min(min_val, img.data[idx]);
                }
            }
            result.data[y * img.width + x] = min_val;
        }
    }
    return result;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./morphology input.png output.png (dilate|erode)\n";
        return 1;
    }
    Image img = load_image(argv[1]);
    Image result;
    if (std::string(argv[3]) == "dilate") {
        result = dilate(img);
    }
    else if (std::string(argv[3]) == "erode") {
        result = erode(img);
    }
    else {
        std::cerr << "Invalid operation. Use either 'dilate' or 'erode'\n";
        return 1;
    }
    save_image(result, argv[2]);
    return 0;
}