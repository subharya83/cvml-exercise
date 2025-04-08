# Camera Pose Estimation from Video

This project estimates 6DoF (6 Degrees of Freedom) camera poses from video frames using feature-based methods.The core theory behind this camera pose estimation is based on the **epipolar geometry** of two views. The system estimates camera motion by analyzing how feature points move between consecutive video frames. For a calibrated camera (where its internal optics are known), matching points between two frames reveal geometric constraints about the camera's movement. The key mathematical object is called the essential matrix, which encodes the relationship between the camera's rotation and translation based on these point correspondences. This matrix is computed robustly using RANSAC to handle outliers from incorrect feature matches.

From the essential matrix, we decompose it to extract the camera's rotation (as a 3D orientation matrix) and translation (as a 3D direction vector). Since we can't determine the absolute distance traveled from images alone (the scale ambiguity problem), the system includes heuristic methods to maintain consistent scale across frames. The final pose is refined by ensuring the estimated movement aligns well with the observed feature movements in the image, and temporal smoothing is applied to reduce jitter in the estimates. This approach effectively builds up the camera's trajectory frame by frame through visual analysis of how the scene appears to change from different viewpoints.


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

### 5. Pose Refinement 

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

## Analysis

### Visualizaion
#### Comparing given pose files [video_a](output/poses_a.jsonl) and [video_b](output/poses_b.jsonl)
![](./assets/original.png)

#### Comparing given pose files [video_a](output/poses_a.jsonl) and [basic pose estimation video_b](output/poses_b_basic.jsonl)
![](./assets/basic.png)

#### Comparing given pose files [video_a](output/poses_a.jsonl) and [enhanced pose estimation video_b](output/poses_b_enhanced.jsonl)
![](./assets/enhanced.png)


### Computational complexity 

The following provide an idea on the runtime of both the basic version and 
the enhanced version

#### Basic version
```shell
$ g++ -std=c++11 cpe.cpp -o cpe `pkg-config --cflags --libs opencv4`
$ time ./cpe input/video_b.mp4 output/poses_b.jsonl
Video info:
  FPS: 6
  Reported frames: 43
  Calculated duration: 7.16667s
  Expected frames (6fps × 7.17s): 43.02
Processed 43 frames. Results written to output/poses_b.jsonl

real	0m2.819s
user	0m5.907s
sys	0m1.022s
```

#### Enhanced version
```shell
$g++ -std=c++11 cpe.cpp -DENHANCE -o cpe `pkg-config --cflags --libs opencv4`
$./cpe input/video_b.mp4 output/poses_b.jsonl
Video info:
  FPS: 6
  Reported frames: 43
  Calculated duration: 7.16667s
  Expected frames (6fps × 7.17s): 43.02
Processed 43 frames. Results written to output/poses_b.jsonl

real	0m52.435s
user	6m14.840s
sys	0m11.409s
```
