#pragma once
#include "image_utils.h"
#include <vector>
#include <cmath>

// Declaration of HOG computation function
std::vector<float> compute_hog(const Image& img, int cell_size = 8, int bin_count = 9);

#ifdef USE_SIFT
// Declaration of gradient orientation function
float get_gradient_orientation(const Image& Ix, const Image& Iy, int x, int y);
#endif