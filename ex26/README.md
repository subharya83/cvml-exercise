# Logo Tracker: A Comparative Study in Python and C++

Welcome to the **Logo Tracker** project! This repository provides an implementation of a logo detection and tracking system in both **Python** and **C++**. The project is designed to be a practical resource for computer vision graduate students who want to explore the differences between Python and C++ in the context of real-time object tracking. Whether you're interested in feature-based detection (ORB) or deep learning-based approaches (Lightweight DETR), this project has something for you.

---

## 🚀 **Project Overview**

The goal of this project is to detect and track a logo in a video stream. The system uses two main approaches:
1. **ORB (Oriented FAST and Rotated BRIEF)**: A feature-based detection method that matches keypoints between a reference logo image and the video frames.
2. **Lightweight DETR**: A deep learning-based approach using a pre-trained DETR (DEtection TRansformer) model for logo detection.

The Python implementation is designed for quick prototyping and ease of use, while the C++ version is optimized for performance and minimal dependencies. Both versions use the **SORT (Simple Online and Realtime Tracking)** algorithm for tracking detected logos across frames.

---

## 📂 **Repository Structure**

```
├── CMakeLists.txt
├── download.sh
├── input
├── logoTracker.py
├── main.cpp
├── README.md
├── sort.cpp
├── sort.h
├── sort.py
└── weights
    ├── yolov3.cfg
    └── yolov3.weights
```

---

## 🛠️ **Features**

### **Python Version**
- **ORB-based logo detection**: Uses OpenCV's ORB implementation for feature matching.
- **Lightweight DETR**: Integrates a pre-trained DETR model for logo detection.
- **SORT tracker**: Tracks detected logos across frames using the SORT algorithm.
- **Easy to use**: Designed for quick prototyping and experimentation.

### **C++ Version**
- **ORB-based logo detection**: Uses OpenCV's ORB implementation for feature matching.
- **SORT tracker**: Implements the SORT algorithm with minimal dependencies.
- **High performance**: Optimized for real-time tracking with efficient memory management.
- **Minimal dependencies**: Only requires OpenCV and Eigen for matrix operations.

---

## 🧠 **Why This Project?**

As a computer vision graduate student, you're likely familiar with Python's ease of use and rapid prototyping capabilities. However, when it comes to deploying real-time systems, C++ is often the language of choice due to its performance and efficiency. This project provides a unique opportunity to:
- **Compare Python and C++ implementations** of the same computer vision task.
- **Understand the trade-offs** between ease of development and performance.
- **Experiment with different tracking algorithms** (e.g., SORT) and detection methods (e.g., ORB, DETR).

---

## 🚀 **Getting Started**

### **Python Version**
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the script**:
   ```bash
   python logoTracker.py -v <video_path> -i <logo_image_path> -o <output_video_path> -a <algorithm>
   ```
   - `<algorithm>`: `0` for ORB, `1` for Lightweight DETR.

### **C++ Version**
1. **Build the project**:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```
2. **Run the executable**:
   ```bash
   ./LogoTracker <video_path> <logo_image_path> <output_video_path>
   ```

---

## Computational workflow

### Classic Feature matching based
```
+-------------------+       +-------------------+       +-------------------+
|                   |       |                   |       |                   |
|  Input Video      | -->   |  ORB Feature      | -->   |  Feature Matching |
|  Frame            |       |  Detection        |       |  (BFMatcher)      |
|                   |       |                   |       |                   |
+-------------------+       +-------------------+       +-------------------+
                                                                |
                                                                v
+-------------------+       +-------------------+       +-------------------+
|                   |       |                   |       |                   |
|  SORT Tracker     | <--   |  Bounding Box     | <--   |    Homography     |
|  (Tracking)       |       |  Estimation       |       |    Calculation    |
|                   |       |                   |       |                   |
+-------------------+       +-------------------+       +-------------------+
        |                                                  
        v                                                  
+-------------------+       +-------------------+ 
|  Tracked Logo     |       |  Output Video     | 
|  Positions        | -->   |  with Bounding    | 
|                   |       |  Boxes            |  
+-------------------+       +-------------------+ 

```
### DETR based

```
+-------------------+       +-------------------+       +-------------------+
|                   |       |                   |       |                   |
|  Input Video      | -->   |  Frame            | -->   |  DETR Model       |
|  Frame            |       |  Preprocessing    |       |  Inference        |
|                   |       |                   |       |                   |
+-------------------+       +-------------------+       +-------------------+
        |                           |                           |
        v                           v                           v
+-------------------+       +-------------------+       +-------------------+
|                   |       |                   |       |                   |
|  Bounding Box     | -->   |  SORT Tracker     | -->   |  Tracked Logo     |
|  Extraction       |       |  (Tracking)       |       |  Positions        |
|                   |       |                   |       |                   |
+-------------------+       +-------------------+       +-------------------+
        |                                                   |
        v                                                   v
+-------------------+                               +-------------------+
|                   |                               |                   |
|  Output Video     | <--                           |  Tracked Logo     |
|  with Bounding    |                               |  Positions        |
|  Boxes            |                               |                   |
+-------------------+                               +-------------------+
```

## 📊 **Performance Comparison**

| Feature                | Python Version          | C++ Version            |
|------------------------|-------------------------|------------------------|
| **Ease of Use**        | High                   | Moderate               |
| **Performance**        | Slower (interpreted)   | Faster (compiled)      |
| **Dependencies**       | OpenCV, PyTorch, ONNX  | OpenCV, Eigen          |
| **Real-Time Tracking** | Possible with optimizations | Better suited for real-time |

---

## 🧪 **Experiments to Try**

1. **Compare ORB and DETR**:
   - How does the accuracy of ORB compare to DETR in different lighting conditions?
   - Which method is faster for real-time tracking?

2. **Optimize the C++ version**:
   - Can you improve the performance of the C++ implementation by optimizing the Kalman filter or matrix operations?

3. **Extend the SORT tracker**:
   - Implement additional features like occlusion handling or re-identification.

---

## 📚 **Resources**

- **OpenCV Documentation**: [https://docs.opencv.org/](https://docs.opencv.org/)
- **Eigen Documentation**: [https://eigen.tuxfamily.org/](https://eigen.tuxfamily.org/)
- **SORT Paper**: [https://arxiv.org/abs/1602.00763](https://arxiv.org/abs/1602.00763)
- **DETR Paper**: [https://arxiv.org/abs/2005.12872](https://arxiv.org/abs/2005.12872)
