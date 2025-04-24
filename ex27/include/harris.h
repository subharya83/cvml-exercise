#pragma once
#include "image_utils.h"
#include "convolution.h"  // Include the convolution header
#include <vector>
#include <cmath>
#include <algorithm>
#include <utility>       // for std::pair
#include <iostream>
#include <fstream>

// Forward declaration (if needed for other headers)
std::vector<std::pair<int, int>> detect_harris_corners(const Image& img, float k = 0.04, float threshold = 0.01);

// Implementation moved to harris.cpp to avoid multiple definition errors
// Only keep declarations in the header file

#ifdef USE_SIFT
// Multi-scale Harris corner detection
std::vector<std::pair<int, int>> detect_harris_corners_multiscale(const Image& img, 
                                                                const std::vector<float>& scales = {0.5f, 1.0f, 2.0f});
#endif