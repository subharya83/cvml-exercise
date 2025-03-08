# Intelligent Image Stitcher

## Applications in Computer Vision Research

This image stitching toolkit provides a robust solution for creating panoramic mosaics with intelligent preprocessing and validation. It's especially useful for:

### 1. Multi-view Reconstruction
Generate high-quality panoramas from drone footage, satellite imagery, or autonomous vehicle cameras. The validation system ensures only relevant imagery from the same scene is included, preventing artifacts in the final reconstruction.

### 2. Biomedical Imaging
Create comprehensive tissue sample mosaics from microscope imagery by automatically filtering out slide preparation errors, contamination, or out-of-focus images before stitching.

### 3. Archaeological Documentation
Document archaeological sites with consistent panoramic imagery by ensuring lighting conditions and camera settings remain relatively constant across the captured images.

### 4. Automated Visual Inspection
Build comprehensive visual records of large industrial equipment or production lines while filtering out unrelated imagery that may have been accidentally included in the dataset.

### 5. Training Data Generation
Create clean panoramic datasets for training deep learning models by automatically removing outliers and inconsistent images.

## Overview

This toolkit employs robust validation techniques based on color distribution and edge density to filter out images that don't belong to a set before stitching them into a panoramic mosaic. The intelligent filtering system dramatically improves stitching quality for imperfect datasets.

The toolkit is available in both Python and C++, allowing for integration into various computational environments and workflows.

## Features

- **Automatic Image Validation**: Uses color histograms and edge density metrics to identify and remove outlier images
- **Reference-Based Filtering**: Automatically establishes reference features from a subset of images
- **Configurable Thresholds**: Fine-tune the filtering sensitivity to match your specific dataset characteristics
- **OpenCV Integration**: Leverages OpenCV's stitching module for high-quality panorama creation
- **Cross-Platform**: Available in both Python and C++ with identical functionality

## Prerequisites

### For Python version
- Python 3.6+
- OpenCV 4.x
- NumPy

### For C++ version
- C++17 compatible compiler
- OpenCV 4.x
- Standard Template Library (STL)

## Installation

### Python
```bash
pip install opencv-python numpy
```

### C++
```bash
# Ubuntu/Debian
sudo apt install libopencv-dev g++

# Compile the program
g++ -std=c++17 imageStitcher.cpp -o imageStitcher `pkg-config --cflags --libs opencv4`
```

## Usage

### Python
```bash
python3 imageStitcher.py -i input_directory -o output.jpg [--hist-threshold 0.6] [--edge-threshold 0.5]
```

### C++
```bash
./imageStitcher -i=input_directory -o=output.jpg [--hist-threshold=0.6] [--edge-threshold=0.5]
```

## Workflow

***Basic version: 
```
+-----------------------+                       +-----------------------+
|     Main Function     | --------------------> |  Load Images from Dir |
|          (A)          |                       |          (B)          |
+-----------------------+                       +-----------------------+
                                                        |
                                                        v
+-----------------------+                       +-----------------------+
| Validate Images Using | <-------------------- | Calculate Reference   |
| Histogram & Edge      |                       | Features (Histogram & |
| Density Comparison (D)|                       | Edge Density) (C)     |
+-----------------------+                       +-----------------------+
           |                                          
           v                                          
+-----------------------+                       +-----------------------+
| Stitch Valid Images   | --------------------> | Save Stitched Panorama|
| Using OpenCV Stitcher |                       | to Output File (F)    |
|          (E)          |                       |                       |
+-----------------------+                       +-----------------------+
```


***Advanced version: 

```
+-------------------+       +-------------------+       +-------------------+
|  Load Images      | ----> |  Scene            | ----> |  Calculate        |
|  from Input Dir   |       |  Classification   |       |  Reference        |
|                   |       |  (Optional)       |       |  Features         |
|  [load_images]    |       |  [SceneClass]     |       |  [Validator]      |
+-------------------+       +-------------------+       +-------------------+
                                                                 |
                                                                 v
+-------------------+       +-------------------+       +-------------------+
|  Stitch Images    | <---- |  Filter Images    | <---- |  Validate Images  |
|  into Panorama    |       |  Based on Scene   |       |  using Selected   |
|  [stitch_images]  |       |  Classification   |       |  Validation Method|
|                   |       |  [SceneClass]     |       |  [Validator]      |
+-------------------+       +-------------------+       +-------------------+
            |                                                  
            v                                                   
+-------------------+       +-------------------+       +-------------------+
|  Save Stitched    | ----> |  Output Panorama  | ----> |  End              |
|  Image to Output  |       |  Image            |       |                   |
|  [cv2.imwrite]    |       |                   |       |                   |
+-------------------+       +-------------------+       +-------------------+
```
## Parameter Tuning

- **Histogram Threshold** (default: 0.6): Controls how similar the color distribution must be between images. Higher values enforce stricter color consistency.
- **Edge Threshold** (default: 0.5): Controls how similar the edge structure must be between images. Higher values enforce stricter structural consistency.

## Technical Details

### Validation Metrics

The validation system uses two complementary metrics:

1. **Color Histogram Comparison**: 
   - Captures the overall color distribution of the image
   - Effective for detecting images from different scenes or lighting conditions
   - Uses correlation-based comparison for robustness to exposure differences

2. **Edge Density Analysis**:
   - Quantifies the structural complexity of the image
   - Helps identify out-of-focus images or those with significantly different content
   - Normalized to account for different image sizes

## Key Enhancements 

1. **SIFT Feature Matching**:
   - SIFT-based image validation using keypoint descriptors
   - Uses ratio test for robust feature matching
   - Calculates match percentage to determine image similarity


2. **Deep Learning Integration**:
   - ResNet18 and MobileNet models for feature extraction
   - cosine similarity metric for comparing deep features

3. **Scene Classification Preprocessing**:
   - Uses a Places365 dataset-trained model to classify scene types
   - Filters out images that don't match the dominant scene category
   - Configurable threshold for determining scene homogeneity

```bash
# Use SIFT features for validation
python imageStitcherPlus.py -i input_directory -o output.jpg --validation sift --match-threshold 0.15

# Use deep learning features with ResNet18
python imageStitcherPlus.py -i input_directory -o output.jpg --validation deep --model resnet18

# Enable scene classification preprocessing with deep learning validation
python imageStitcherPlus.py -i input_directory -o output.jpg --validation deep --scene-classification --scene-threshold 0.6

# Use MobileNet for both scene classification and feature validation
python imageStitcherPlus.py -i input_directory -o output.jpg --validation deep --model mobilenet --scene-classification
```

## Future Improvements

Potential enhancements for research purposes:

- Adaptive threshold selection based on dataset characteristics
- Multi-band blending for improved seam quality
- GPU acceleration for large dataset processing

