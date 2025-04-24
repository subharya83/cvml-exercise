# Computer Vision Problems in C++ with minimal or no third-party dependencies

A list of problems in computer vision that can be explicitly solved leveraging standard libraries 
(like `<algorithm>`, `<vector>`, `<opencv>` for basic matrix ops if allowed, or even raw arrays). 

---

### 1. **Image Convolution (Gaussian Blur, Edge Detection)**
   - **Problem**: Apply a kernel (e.g., 3×3 Gaussian blur or Sobel edge detection) to an image.
   - **Minimal Dependencies**: Raw pixel arrays (no OpenCV needed if you handle I/O manually).
   - **Solution**:
     - Represent the image as a 2D array (or flattened `std::vector`).
     - Manually implement kernel convolution with boundary checks.
     - Optimize with loop unrolling or SIMD (if performance is critical).

---

### 2. **Connected Components Labeling (Binary Image)**
   - **Problem**: Label connected regions in a binary image (e.g., for object detection).
   - **Minimal Dependencies**: None (just C++ STL).
   - **Solution**:
     - Use a **union-find (disjoint set)** data structure to merge labels.
     - Implement the **Two-Pass Algorithm**:
       1. First pass: Assign provisional labels and record equivalences.
       2. Second pass: Resolve equivalences and relabel.

---

### 3. **Optical Flow (Lucas-Kanade Method)**
   - **Problem**: Estimate motion between two consecutive frames.
   - **Minimal Dependencies**: Eigen (for small matrix math) or hand-rolled linear algebra.
   - **Solution**:
     - Compute spatial/temporal gradients over a window.
     - Solve the least-squares problem: `A^T A d = A^T b` for pixel motion `d`.
     - Use `<cmath>` for gradient calculations.

---

### 4. **Perspective Transform (Homography Estimation)**
   - **Problem**: Warp an image using a 3×3 homography matrix.
   - **Minimal Dependencies**: Eigen or manual matrix inversion (4×4 SVD is tractable).
   - **Solution**:
     - Use **Direct Linear Transform (DLT)** with 4 point correspondences.
     - Implement bilinear interpolation for smooth warping.

---

### 5. **K-Means Clustering (Color Quantization)**
   - **Problem**: Reduce the number of colors in an image.
   - **Minimal Dependencies**: None (STL only).
   - **Solution**:
     - Represent pixels as 3D (RGB) vectors.
     - Iteratively update cluster centers and reassign pixels.

---

### 6. **Harris Corner Detection**
   - **Problem**: Detect corners in an image.
   - **Minimal Dependencies**: `<cmath>` for gradients and non-max suppression.
   - **Solution**:
     - Compute structure tensor (gradient products `I_x², I_y², I_x I_y`).
     - Calculate corner response: `R = det(M) - k·trace(M)²`.
     - Threshold and suppress non-maxima.

---

### 7. **RANSAC for Outlier Rejection**
   - **Problem**: Fit a model (e.g., line, homography) to noisy data.
   - **Minimal Dependencies**: `<random>` for sampling, `<vector>` for data.
   - **Solution**:
     - Randomly sample minimal subsets.
     - Score inliers using a threshold (e.g., reprojection error).

---

### 8. **Histogram of Oriented Gradients (HOG)**
   - **Problem**: Compute HOG features for object detection.
   - **Minimal Dependencies**: `<cmath>` for gradients and histogram bins.
   - **Solution**:
     - Compute gradients (`Gx`, `Gy`) via finite differences.
     - Bin orientations into cells (weighted by gradient magnitude).

---

### 9. **Image Morphology (Erosion/Dilation)**
   - **Problem**: Apply binary morphology operations.
   - **Minimal Dependencies**: None.
   - **Solution**:
     - Use a sliding window to compute min/max over a kernel (structuring element).

---

### 10. **Brute-Force Feature Matching**
   - **Problem**: Match keypoints between two images (e.g., SIFT-like features).
   - **Minimal Dependencies**: `<vector>`, `<algorithm>` for distance sorting.
   - **Solution**:
     - Compute descriptors (e.g., normalized patches).
     - Compare using SSD/SAD or cosine distance.

---

## Compiling/Building instructions

