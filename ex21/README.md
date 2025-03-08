# Image Alignment Tool

This repository contains two implementations of an image alignment tool: one in **Python** (`alignImages.py`) and one in **C++** (`alignImages.cpp`). These tools are designed to align a target image to a reference image, which is particularly useful in applications like **satellite image analysis**, medical imaging, and computer vision.

---

## Features

- **Two Algorithms**:
  - **SIFT-based Homography**: Robust alignment for complex transformations (rotation, scaling, perspective).
  - **Phase Correlation**: Fast and efficient alignment for pure translations.
- **Command-Line Interface**: Easy-to-use command-line arguments for specifying inputs and outputs.
- **Translation Parameters**: Returns the translation parameters (\( t_x, t_y \)) in pixels.
- **Cross-Platform**: Works on any platform with Python or C++ and OpenCV installed.

---

## Use Case: Satellite Image Analysis

Satellite images are often captured at different times, angles, or resolutions. Aligning these images is crucial for:

1. **Change Detection**:
   - Align images taken at different times to detect changes in land use, deforestation, or urban development.
   - Example: Monitoring the growth of a city over a decade.

2. **Image Fusion**:
   - Combine images from different sensors (e.g., optical and infrared) to create a composite image with enhanced details.
   - Example: Merging high-resolution optical images with thermal data for better analysis.

3. **Disaster Management**:
   - Align pre- and post-disaster images to assess damage and plan recovery efforts.
   - Example: Comparing satellite images before and after a hurricane to identify affected areas.

4. **Environmental Monitoring**:
   - Track changes in glaciers, forests, or water bodies over time.
   - Example: Monitoring the retreat of glaciers due to climate change.

---

## Computational workflow
```
+-------------------+       +-------------------+       +-------------------+
|                   |       |                   |       |                   |
|  Command Line     |       |  Load Images      |       |  Check Image      |
|  Argument Parsing | ----> |  (Reference &     | ----> |  Loading Success  |
|                   |       |  Target)          |       |                   |
+-------------------+       +-------------------+       +-------------------+
                                                            |
                                                            V
                                          +-------------------+       
                                          |                   |  
                                          |  Select Algorithm | 
                                          |  (0: SIFT, 1:     | 
                                          | Phase Correlation)|  
                                          +-------------------+       
                                             |              |
                                             v              v
                           +-------------------+       +-------------------+
                           |                   |       |                   |
                           |  SIFT-Based       |       |  Phase Correlation|
                           |  Alignment        |       |  Alignment        |
                           |                   |       |                   |
                           +-------------------+       +-------------------+
                                |                           |
                                v                           v
+-------------------+       +-------------------+       +-------------------+
|                   |       |                   |       |                   |
|  Compute          |       |  Detect & Match   |       |  Compute          |
|  Homography       | <---- |  Keypoints        |       |  Translation      |
|  (RANSAC)         |       |                   |       |  (Phase Shift)    |
+-------------------+       +-------------------+       +-------------------+
                                             |             |
                                             v             v
                                          +-------------------+
                                          |                   |
                                          |  Print Translation|
                                          |  Parameters       |
                                          |                   |
                                          +-------------------+


+-------------------+       +-------------------+       +-------------------+
|                   |       |                   |       |                   |
|  Warp Target      |       |  Save Aligned     |       |  Print Translation|
|  Image            | ----> |  Image            | ----> |  Parameters       |
|                   |       |                   |       |                   |
+-------------------+       +-------------------+       +-------------------+

```

## Python Implementation (`alignImages.py`)

### Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- scikit-image (`scikit-image`)
- numpy (`numpy`)

Install the dependencies:
```bash
pip install opencv-python scikit-image numpy
```

### Usage

```bash
python alignImages.py -r reference_image.png -t target_image.png -o aligned_image.png -a <algorithm>
```

- `-r`: Path to the reference image.
- `-t`: Path to the target image.
- `-o`: Path to save the aligned image.
- `-a`: Algorithm selection (`0` for SIFT, `1` for phase correlation).

#### Example

```bash
python alignImages.py -r reference.png -t target.png -o aligned.png -a 0
```

---

## C++ Implementation (`alignImages.cpp`)

### Requirements

- OpenCV 4.x
- C++ compiler (e.g., g++)

### Compilation

```bash
g++ -std=c++11 -o alignImages alignImages.cpp `pkg-config --cflags --libs opencv4`
```

### Usage

```bash
./alignImages -r reference_image.png -t target_image.png -o aligned_image.png -a <algorithm>
```

- `-r`: Path to the reference image.
- `-t`: Path to the target image.
- `-o`: Path to save the aligned image.
- `-a`: Algorithm selection (`0` for SIFT, `1` for phase correlation).

#### Example

```bash
./alignImages -r reference.png -t target.png -o aligned.png -a 1
```

---

## Output

- **Aligned Image**: Saved to the specified output path.
- **Translation Parameters**: Printed to the console in the format:
  ```
  Translation parameters (in pixels): (tx, ty) = (50.00, 30.00)
  ```

---

## Example: Satellite Image Alignment

### Input Images

1. **Reference Image** (`reference.png`):
   ![Reference Image](https://via.placeholder.com/500x500?text=Reference+Image)

2. **Target Image** (`target.png`):
   ![Target Image](https://via.placeholder.com/500x500?text=Target+Image)

### Command

```bash
python alignImages.py -r reference.png -t target.png -o aligned.png -a 0
```
