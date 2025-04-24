#pragma once
#include "image_utils.h"
#include <Eigen/Dense>
#include <vector>

// Declaration of Lucas-Kanade optical flow function
void lucas_kanade(const Image& img1, const Image& img2, int window_size);