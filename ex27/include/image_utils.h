#pragma once

#include <vector>
#include <string>
#include <stdexcept>  // for std::runtime_error
#include <algorithm>  // for std::min, std::max
#include <cstdint>    // for uint8_t, uint32_t

// STB image libraries
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Grayscale image type
struct Image {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> data;

    // Helper methods
    bool empty() const { return data.empty(); }
    size_t size() const { return width * height; }
    uint8_t& at(int x, int y) { return data[y * width + x]; }
    const uint8_t& at(int x, int y) const { return data[y * width + x]; }
};

// RGB image type
struct ImageRGB {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> data; // interleaved RGB (size = width*height*3)

    bool empty() const { return data.empty(); }
    size_t size() const { return width * height * 3; }
};

// Load grayscale image
inline Image load_image(const std::string& path) {
    int w, h, c;
    uint8_t* img = stbi_load(path.c_str(), &w, &h, &c, 1); // Force grayscale
    if (!img) throw std::runtime_error("Failed to load image: " + path);
    Image result{w, h, std::vector<uint8_t>(img, img + w * h)};
    stbi_image_free(img);
    return result;
}

// Load RGB image
inline ImageRGB load_image_rgb(const std::string& path) {
    int w, h, c;
    uint8_t* img = stbi_load(path.c_str(), &w, &h, &c, 3); // Force RGB
    if (!img) throw std::runtime_error("Failed to load image: " + path);
    ImageRGB result{w, h, std::vector<uint8_t>(img, img + w * h * 3)};
    stbi_image_free(img);
    return result;
}

// Save grayscale image
inline void save_image(const Image& img, const std::string& path) {
    if (img.empty()) throw std::runtime_error("Empty image cannot be saved");
    stbi_write_png(path.c_str(), img.width, img.height, 1, img.data.data(), img.width);
}

// Save RGB image
inline void save_image_rgb(const ImageRGB& img, const std::string& path) {
    if (img.empty()) throw std::runtime_error("Empty image cannot be saved");
    stbi_write_png(path.c_str(), img.width, img.height, 3, img.data.data(), img.width * 3);
}

// Convert RGB to grayscale (luminance formula)
inline Image rgb_to_grayscale(const ImageRGB& rgb) {
    Image gray{rgb.width, rgb.height, std::vector<uint8_t>(rgb.width * rgb.height)};
    
    for (int i = 0; i < rgb.width * rgb.height; i++) {
        const uint8_t* pixel = &rgb.data[i * 3];
        gray.data[i] = static_cast<uint8_t>(
            0.299f * pixel[0] + 0.587f * pixel[1] + 0.114f * pixel[2]
        );
    }
    return gray;
}

// Threshold image
inline Image threshold(const Image& img, uint8_t thresh) {
    Image result = img;
    for (auto& p : result.data) {
        p = (p > thresh) ? 255 : 0;
    }
    return result;
}

// Invert image
inline Image invert(const Image& img) {
    Image result = img;
    for (auto& p : result.data) {
        p = 255 - p;
    }
    return result;
}

// Check if point is within image bounds
inline bool in_bounds(const Image& img, int x, int y) {
    return x >= 0 && x < img.width && y >= 0 && y < img.height;
}