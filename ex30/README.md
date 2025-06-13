# Point Cloud Registration Exercise: From Data Generation to Alignment

## Overview

This repository provides a hands-on exercise in point cloud registration, covering the complete pipeline 
from synthetic data generation to alignment using state-of-the-art algorithms. The implementation showcases:

1. **Synthetic Data Generation**: Create customizable 3D polygonal contours with controlled noise parameters
2. **Multiple Registration Methods**: Compare different alignment techniques including ICP, FGR, and hybrid approaches
3. **Metrics-Driven Evaluation**: Quantitative assessment of registration quality with multiple metrics
4. **Visualization**: 3D visualization of registration results

## Why This Exercise Matters

For computer vision graduate students and practitioners, this exercise offers:

- **Practical implementation** of fundamental point cloud registration algorithms
- **Controlled experimentation** with synthetic data where ground truth is known
- **Performance comparison** between different registration approaches
- **Visual feedback** to intuitively understand algorithm behavior
- **Metrics-based evaluation** to quantitatively assess results

## Key Features

### Data Generation (`datagen.py`)

- Generates irregular polygonal contours in 3D space
- Control over:
  - Number of points
  - Planar deviation (z-tolerance)
  - Size (radius)
  - Position (center coordinates)
- CSV export for interoperability
- Optional 3D visualization

### Registration Pipeline (`register_pointsets.py`)

- Implements multiple registration methods:
  - **ICP (Iterative Closest Point)**: Classic point-to-point alignment
  - **FGR (Fast Global Registration)**: Feature-based coarse alignment
  - **FGR+ICP**: Hybrid approach combining both methods
  - **Multi-scale ICP**: Hierarchical registration with decreasing voxel sizes
- Computes comprehensive metrics:
  - Fitness score
  - Inlier RMSE
  - Mean/median point distances
- Visualization of source, target, and registered point sets

## Getting Started

1. Generate sample data:
   ```bash
   python datagen.py 100 80 0.05 contour1.csv contour2.csv --plot
   ```

2. Run registration:
   ```bash
   python register_pointsets.py contour1.csv contour2.csv --method fgr+icp --visualize
   ```

## Experiment Ideas

1. **Noise Analysis**: Vary the z-tolerance in data generation and observe registration accuracy
2. **Point Density**: Test with different numbers of points in each contour
3. **Algorithm Comparison**: Run all methods on the same data and compare metrics
4. **Initial Pose**: Modify the initial positions in datagen.py to test robustness
5. **Parameter Tuning**: Experiment with different ICP parameters and FGR thresholds

