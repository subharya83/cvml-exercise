#pragma once
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <vector>
#include <string>

// Grayscale image type
struct Image {
    int width, height;
    std::vector<uint8_t> data;
};

Image load_image(const std::string& path) {
    int w, h, c;
    uint8_t* img = stbi_load(path.c_str(), &w, &h, &c, 1); // Force grayscale
    if (!img) throw std::runtime_error("Failed to load image");
    Image result{w, h, std::vector<uint8_t>(img, img + w * h)};
    stbi_image_free(img);
    return result;
}

void save_image(const Image& img, const std::string& path) {
    stbi_write_png(path.c_str(), img.width, img.height, 1, img.data.data(), img.width);
}