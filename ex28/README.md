# Camera Pose Estimation from Video

This project estimates 6DoF (6 Degrees of Freedom) camera poses from video frames using feature-based methods.
The core theory behind this camera pose estimation is based on the **epipolar geometry** of two views. For a calibrated 
camera (known intrinsic parameters), the essential matrix \( E \) relates corresponding points \( x \) and \( x' \) in 
normalized image coordinates through the epipolar constraint:

\[
x'^\top E x = 0
\]

where \( E = [t]_\times R \) contains the relative rotation \( R \) and translation \( t \) between views (with \( [t]_\times \) being the skew-symmetric matrix of \( t \)). The 8-point algorithm or RANSAC-based methods solve for \( E \) from point correspondences. The pose \( (R, t) \) is then recovered via SVD decomposition of \( E \), enforcing the cheirality constraint to ensure physically valid solutions. For monocular sequences, the translation is recovered only up to scale (\( \|t\| = 1 \)), necessitating heuristic scale estimation (e.g., \( t_{new} = \alpha t \) based on feature motion magnitude). The reprojection error \( \sum_i \|x'_i - K(RX_i + t)\|^2 \) (where \( K \) is the camera matrix) is minimized during refinement, typically via PnP (Perspective-n-Point) algorithms like RANSAC or Levenberg-Marquardt optimization.

## Key Computational Blocks

### 1. Feature Detection and Extraction

**Basic Version (ORB):**
```cpp
Ptr<ORB> orb = ORB::create();
orb->detectAndCompute(gray, noArray(), keypoints, descriptors);
```
- Uses ORB (Oriented FAST and Rotated BRIEF) features
- Fast but less accurate for pose estimation
- Binary descriptors with Hamming distance matching

**Enhanced Version (SIFT):**
```cpp
Ptr<SIFT> detector = SIFT::create(0, 3, 0.04, 10);
detector->detectAndCompute(gray, noArray(), keypoints, descriptors);
```
- Uses SIFT (Scale-Invariant Feature Transform) features
- More accurate but computationally intensive
- Floating-point descriptors with L2 norm matching

### 2. Feature Matching

**Basic Version:**
```cpp
matcher.match(prevDescriptors, descriptors, matches);
// Simple distance threshold filtering
```
- Basic brute-force matching with distance threshold

**Enhanced Version:**
```cpp
matcher.knnMatch(prevDescriptors, descriptors, knn_matches, 2);
// Ratio test filtering (Lowe's ratio test)
```
- Uses k-NN matching with ratio test
- More robust to false matches
- Typically retains only high-confidence matches

### 3. Essential Matrix Estimation

```cpp
E = findEssentialMat(prevMatchedPoints, matchedPoints, cameraMatrix, 
                    RANSAC, 0.999, 1.0, mask);
```
- Estimates the essential matrix from point correspondences
- Uses RANSAC for robust estimation
- In enhanced version: tighter parameters (0.9999 confidence, 0.5 pixel threshold)

### 4. Pose Recovery

```cpp
recoverPose(E, prevMatchedPoints, matchedPoints, cameraMatrix, R, t, mask);
```
- Decomposes essential matrix to get rotation and translation
- Returns relative camera motion between frames
- Translation is up to scale for monocular sequences

### 5. Pose Refinement (Enhanced Version Only)

**Scale Estimation:**
```cpp
currentScale = estimateScale(prevMatchedPoints, matchedPoints);
t *= currentScale;
```
- Heuristic scale estimation for monocular sequences
- Helps maintain consistent scale across frames

**Temporal Smoothing:**
```cpp
poseHistory.addPose(currentPose);
currentPose = poseHistory.getSmoothedPose();
```
- Maintains a window of previous poses
- Applies moving average smoothing
- Reduces jitter in pose estimates

## Compile

1. **Basic Version:**
   ```bash
   g++ -std=c++11 cpe.cpp -o cpe `pkg-config --cflags --libs opencv4`
   ```

2. **Enhanced Version** (uncomment `#define ENHANCE`):
   ```bash
   g++ -std=c++11 cpe.cpp -DENHANCE=ON -o cpe `pkg-config --cflags --libs opencv4`
   ```
