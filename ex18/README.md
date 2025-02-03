# CUDA Video Processing with Integral Images

This project demonstrates how to process video frames using CUDA to compute integral images, which are widely used in computer vision applications such as object detection, feature extraction, and image filtering. The project also includes HDF5 integration for storing processed frames and PTX code generation for CUDA kernel analysis.

## Overview

The program processes a video frame-by-frame, computes integral images using CUDA, and stores the results in an HDF5 file. Key features include:
- **CUDA-accelerated integral image computation**
- **Efficient memory management with `std::unique_ptr`**
- **HDF5 integration for storing processed frames**
- **PTX code generation for CUDA kernel analysis**
- **Error handling for CUDA, OpenCV, and HDF5 operations**

## Requirements

- **NVIDIA CUDA Toolkit** (minimum version 11.0)
- **NVIDIA GPU** with compute capability 5.2 or higher
- **OpenCV** (for video processing)
- **HDF5 C++ Library** (for storing processed frames)
- **gcc/g++ compiler**
- **Linux/Unix environment** (for provided commands)

## File Structure

```
cuda-video-processing/
├── src/
│   ├── video_processor.cpp
│   ├── cuda_kernels.cu
│   └── cuda_kernels.cuh
├── CMakeLists.txt
├── README.md
└── .gitignore
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cuda-video-processing.git
   cd cuda-video-processing
   ```

2. Install dependencies:
   ```bash
   sudo apt-get install libopencv-dev libhdf5-dev
   ```

3. Verify CUDA installation:
   ```bash
   nvcc --version
   ```

4. Build the project using CMake:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

## Usage

1. Run the executable with a video file and output HDF5 file:
   ```bash
   ./video_processor <video_path> <output_h5_path>
   ```

2. View the generated HDF5 file:
   ```bash
   h5dump output.h5
   ```

## PTX Code Generation

To generate PTX (Parallel Thread Execution) code for CUDA kernel analysis, use the following command:
```bash
nvcc -ptx src/cuda_kernels.cu -o cuda_kernels.ptx
```

You can inspect the generated PTX code to understand how the CUDA kernels are compiled and optimized:
```bash
cat cuda_kernels.ptx
```

## Code Explanation

### CUDA Kernels
The project includes three CUDA kernels:
1. **`convertToFloat`**: Converts grayscale frames from `unsigned char` to `float`.
2. **`horizontalScan`**: Computes the horizontal prefix sum for integral image computation.
3. **`verticalScan`**: Computes the vertical prefix sum to complete the integral image.

Each kernel is optimized for parallel execution and uses shared memory for efficient data access.

### Memory Management
The project demonstrates proper CUDA memory handling:
- **Host memory allocation** using `std::vector` and `cv::Mat`.
- **Device memory allocation** using `cudaMalloc` and managed by `std::unique_ptr`.
- **Memory transfers** using `cudaMemcpyAsync` for overlapping computation and data transfer.

### Execution Configuration
The kernels are launched with optimized grid and block sizes:
```cuda
dim3 blockSize(16, 16);
dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
              (height + blockSize.y - 1) / blockSize.y);
```

### HDF5 Integration
Processed frames are stored in an HDF5 file with chunking and compression for efficient storage:
```cpp
H5::DataSet dataset = file->createDataSet(
    "integral_images", H5::PredType::NATIVE_FLOAT, dataspace, plist);
```

## Performance Considerations

- **Block Size**: The example uses a block size of 16x16 threads. Experiment with different block sizes for optimal performance.
- **Shared Memory**: The kernels use shared memory for efficient prefix sum computation.
- **Streams**: Asynchronous memory transfers and kernel execution are used to overlap computation and data transfer.
- **Chunking and Compression**: HDF5 datasets are configured with chunking and GZIP compression for efficient storage.

## Example Use Cases

1. **Object Detection**: Integral images are used in algorithms like Viola-Jones for efficient feature computation.
2. **Image Filtering**: Integral images enable fast computation of box filters and other convolution operations.
3. **Feature Extraction**: Integral images are used in HOG (Histogram of Oriented Gradients) and other feature extraction methods.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.
