# Dead Pixel Detection - SPADE (Statistical Pixel Anomaly Detection Engine)

A robust tool for detecting dead, stuck, or malfunctioning pixels in video footage using statistical analysis and computer vision techniques.

## Problem Design

### Objectives
- Detect dead or malfunctioning pixels in video footage
- Track pixel defects across multiple frames
- Provide confidence scores for detected defects
- Generate detailed reports of findings
- Support real-time visualization of detected defects

### Technical Approach
The solution employs a sophisticated statistical approach:
1. Gradient Analysis: Computes first and second-order gradients using 8-directional difference matrices
2. Statistical Detection: Uses Mahalanobis distance to identify anomalous pixel behavior
3. Temporal Tracking: Monitors pixel defects across consecutive frames to filter out noise
4. Confidence Scoring: Assigns confidence levels based on statistical deviation from normal behavior

### Constraints
- Requires sufficient contrast in video content for reliable detection
- Processing speed depends on video resolution and hardware capabilities
- Minimum persistence threshold to filter out temporary artifacts
- Border pixels require special handling due to edge effects

## Data Preparation

### Input Requirements
- Video files in formats supported by OpenCV
- Sufficient frame rate for temporal analysis
- Adequate lighting and contrast in source material

### Testing Data Generation
The project includes a `VideoCorruptor` tool that:
- Introduces controlled pixel defects at specified rates
- Creates contiguous defect regions
- Simulates various types of pixel failures
- Generates test videos with known ground truth

### Data Processing Pipeline
1. Frame Extraction: Convert video frames to grayscale
2. Gradient Computation: Calculate directional differences
3. Statistical Analysis: Compute covariance matrices and distances
4. Temporal Aggregation: Track defects across frames

## Code Organization

### Core Components
1. **DeadPixelDetector Class**
   - Main detection engine
   - Handles frame processing and analysis
   - Manages defect tracking and reporting

2. **PixelDefect Structure**
   - Stores defect information
   - Tracks temporal persistence
   - Maintains confidence scores

3. **VideoCorruptor Class**
   - Test data generation
   - Configurable corruption parameters
   - Realistic defect simulation

### Implementation Variants
- `Spade.cpp`: Base C++ implementation
- `Spade.py`: Python implementation with additional logging
- `Spadeplus.cpp`: Multi-threaded C++ version for improved performance

## Test Cases

### Functionality Tests
1. **Basic Detection**
   - Single dead pixel detection
   - Multiple defect detection
   - Confidence score accuracy

2. **Temporal Tracking**
   - Persistence verification
   - Frame-to-frame consistency
   - Start/end frame accuracy

3. **Edge Cases**
   - Border pixel handling
   - Low contrast regions
   - High noise environments

### Performance Tests
- Processing speed benchmarks
- Memory usage monitoring
- Multi-threading efficiency
- Different video resolutions

## Further Optimizations and Improvements

### Performance Enhancements
1. **Computational Optimization**
   - GPU acceleration potential
   - Vectorized operations
   - Memory usage optimization
   - Parallel processing improvements

2. **Detection Accuracy**
   - Advanced filtering techniques
   - Machine learning integration
   - Adaptive thresholding
   - Pattern recognition


```shell
# Compile video corruptor, SpatiotemPoral Anomalous pixel DEtector
g++ -std=c++11 -o videoCorruptor videoCorruptor.cpp `pkg-config --cflags --libs opencv4`
g++ -std=c++11 -o Spade Spade.cpp `pkg-config --cflags --libs opencv4`
```

```shell
./Spade -i output/c-hummer.mp4 -o output/c-hummer.csv -v output/o-hummer.mp4
````

### Multi-threaded version improvements

Key Changes:
- Thread Pool: A thread pool is created using std::thread and std::mutex to manage concurrent processing of frames.
- Frame Queue: A queue (std::queue) is used to store frames that need to be processed. Threads will pop frames from this queue and process them.
- Mutexes: std::mutex is used to synchronize access to shared resources like the frame queue and the defects data structures.
- Worker Function: A lambda function (worker) is defined to process frames. Each thread runs this function, which processes frames until the queue is empty.

Notes:
- The number of threads is determined by `std::thread::hardware_concurrency()`, which returns the number of concurrent threads supported by the hardware.
- The `update_defects()`  is protected by a mutex to ensure thread safety when updating the `active_defects_` and `completed_defects_` structures.
- The `visualize_defects()` is also protected by a mutex to ensure thread-safe access to the active_defects_ structure.
