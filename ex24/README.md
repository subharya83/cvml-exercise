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

# macOS
brew install opencv

# Compile the program
g++ -std=c++17 image_stitcher.cpp -o image_stitcher_cpp `pkg-config --cflags --libs opencv4`
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

## How It Works

The program follows these steps to create high-quality panoramas:

1. **Image Loading**: Loads all images from the specified input directory
2. **Reference Selection**: Automatically selects a representative subset of images
3. **Feature Extraction**: Computes color histograms and edge density metrics for all images
4. **Validation**: Compares each image against the reference set to identify outliers
5. **Filtering**: Removes images that don't meet the similarity thresholds
6. **Stitching**: Uses OpenCV's stitcher to create the final panoramic mosaic

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

## Future Improvements

Potential enhancements for research purposes:

- SIFT/SURF-based feature matching for more precise validation
- Deep learning-based scene classification as a preprocessing step
- Adaptive threshold selection based on dataset characteristics
- Multi-band blending for improved seam quality
- GPU acceleration for large dataset processing

