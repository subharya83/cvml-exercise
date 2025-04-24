#pragma once
#include "image_utils.h"
#include <algorithm>

// Declarations
Image dilate(const Image& img, int kernel_size = 3);
Image erode(const Image& img, int kernel_size = 3);