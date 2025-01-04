# Single Object 3D Reconstruction

Create a 3D point cloud reconstruction from video sequences or image sets of objects in clear backgrounds. This project implements Structure from Motion (SfM) techniques using OpenCV to generate colored point clouds from input footage.

## Rough Problem Sketch

Given a video of an object in a relatively clear background where all frames capture various poses of the object, we use computer vision techniques to create a 3D point cloud representation. The implementation provides both C++ and Python solutions using standard computer vision libraries.

### Problem Design

#### Objectives
- Create a 3D point cloud from video/image sequences
- Preserve color information from the original footage
- Support both video files and image directories as input
- Generate industry-standard output formats (.ply)

#### Constraints
- Object should be in a clear, uncluttered background
- Video/images should capture multiple angles of the object
- Sufficient lighting and texture for feature detection
- Static object with camera movement (or vice versa)

#### Success Criteria
- Accurate point cloud generation
- Color preservation in the final model
- Robust feature matching across frames
- Efficient processing of input sequences

### Data Preparation

#### Input Requirements
1. Video Format:
   - Common formats (mp4, avi, etc.)
   - Clear, well-lit footage
   - Steady camera movement
   - Complete object coverage

2. Image Sequence:
   - Supported formats: jpg, jpeg, png, bmp
   - Sequential captures around the object
   - Consistent lighting conditions
   - Sufficient overlap between adjacent images

#### Preprocessing Steps
1. Frame extraction from video (if video input)
2. Image quality assessment
3. Feature detection and matching
4. Pose estimation between consecutive frames

### Code Organization

#### Project Structure
```
3d-reconstruction/
├── cpp/
│   ├── src/
│   │   └── reconstruct.cpp
│   └── CMakeLists.txt
├── python/
│   └── reconstruct.py
└── README.md
```

#### Key Components
1. PointCloudReconstructor Class:
   - Image/video loading
   - Feature detection and matching
   - 3D point triangulation
   - PLY file generation

2. Main Workflow:
   - Command-line argument parsing
   - Input validation
   - Reconstruction pipeline execution
   - Output file generation

### Test Cases

#### Input Validation
- Test video file loading
- Test image directory loading
- Invalid file handling
- Empty directory handling

#### Feature Detection
- Sufficient features found
- Proper matching between frames
- Handling low-texture scenarios
- Outlier rejection

#### Reconstruction Quality
- Point cloud density
- Color accuracy
- Geometric accuracy
- Scale consistency

#### Edge Cases
- Very short videos
- High-speed camera movement
- Varying lighting conditions
- Complex object geometries

### Further Optimizations and Improvements

#### Performance Enhancements
- Parallel processing for feature detection
- GPU acceleration for matching
- Memory optimization for large datasets
- Batch processing capabilities

#### Quality Improvements
- Bundle adjustment implementation
- Dense reconstruction options
- Automatic background removal
- Surface mesh generation

#### Additional Features
- Multiple output format support (.obj, .stl)
- Real-time visualization
- Camera parameter calibration
- Texture mapping

#### User Experience
- GUI implementation
- Progress monitoring
- Error reporting
- Configuration options

## Usage

### C++ Version
```bash
# Build the project
mkdir build && cd build
cmake ..
make

# Run reconstruction
./reconstruct -i <input_path> -o <output_path>
```

### Python Version
```bash
# Run reconstruction
python reconstruct.py -i <input_path> -o <output_path>
```

## Dependencies
- OpenCV (>= 4.0)
- CMake (for C++ version)
- Python 3.6+ (for Python version)
- NumPy
