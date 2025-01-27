You're absolutely correct! Since the C++ implementation relies on **Eigen** and **Boost** libraries (in addition to OpenCV), they should be included in the compilation commands in the `README.md`. Here's the updated `README.md` with the correct compilation commands:

---

# Camera Motion Classification

This project provides tools for classifying camera motion in video sequences. It includes both Python and C++ implementations.

## Table of Contents
1. [Python Implementation](#python-implementation)
2. [C++ Implementation](#cpp-implementation)
3. [Dependencies](#dependencies)
4. [Usage](#usage)
   - [Training the Model](#training-the-model)
   - [Running Inference](#running-inference)
5. [File Descriptions](#file-descriptions)

---

## Python Implementation

### Dependencies
To run the Python implementation, ensure the following Python packages are installed:
- `opencv-python`
- `scipy`
- `numpy`
- `scikit-learn`
- `joblib`

You can install these dependencies using `pip`:
```bash
pip install opencv-python scipy numpy scikit-learn joblib
```

### Usage

#### Training the Model
To train the model using the Python implementation, run the following command:
```bash
python Train.py -v <video_directory> -d <dataset_file> -m <model_file>
```
- `<video_directory>`: Directory containing video files for training.
- `<dataset_file>`: Output file to save the dataset (default: `output/dataset.npz`).
- `<model_file>`: Output file to save the trained model (default: `output/CamMotionModel.pkl`).

#### Running Inference
To classify camera motion in a test video, run the following command:
```bash
python Inference.py -m <model_file> -t <test_video_path>
```
- `<model_file>`: Path to the trained model file (default: `output/CamMotionModel.pkl`).
- `<test_video_path>`: Path to the test video file.

---

## C++ Implementation

### Dependencies
To compile and run the C++ implementation, ensure the following libraries are installed:
- **OpenCV** (for image processing)
- **Eigen** (for matrix operations)
- **Boost** (for serialization, optional for PCA/SVM)

#### Installation on Ubuntu
```bash
sudo apt-get install libopencv-dev libeigen3-dev libboost-all-dev
```

### Compilation
To compile the C++ code, use the following commands. Ensure that the paths to Eigen and Boost are correctly specified in your environment.

#### Compile `CamMotionClassification.cpp`:
```bash
g++ -std=c++17 -o CamMotionClassification CamMotionClassification.cpp -I/usr/include/eigen3 -I/usr/include/boost -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_xfeatures2d
```

#### Compile `Train.cpp`:
```bash
g++ -std=c++17 -o Train Train.cpp -I/usr/include/eigen3 -I/usr/include/boost -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_xfeatures2d
```

#### Compile `Inference.cpp`:
```bash
g++ -std=c++17 -o Inference Inference.cpp -I/usr/include/eigen3 -I/usr/include/boost -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_xfeatures2d
```

### Usage

#### Training the Model
To train the model using the C++ implementation, run the following command:
```bash
./Train -v <video_directory> -d <dataset_file> -m <model_file>
```
- `<video_directory>`: Directory containing video files for training.
- `<dataset_file>`: Output file to save the dataset (default: `output/dataset.npz`).
- `<model_file>`: Output file to save the trained model (default: `output/CamMotionModel.pkl`).

#### Running Inference
To classify camera motion in a test video, run the following command:
```bash
./Inference -m <model_file> -t <test_video_path>
```
- `<model_file>`: Path to the trained model file (default: `output/CamMotionModel.pkl`).
- `<test_video_path>`: Path to the test video file.

---

## File Descriptions

### Python Files
1. **CamMotionClassification.py**: Contains the `CamMotionClassifier` class for feature extraction and dataset creation.
2. **Train.py**: Script to train the camera motion classification model.
3. **Inference.py**: Script to classify camera motion in a test video using the trained model.

### C++ Files
1. **CamMotionClassification.cpp**: C++ implementation of the `CamMotionClassifier` class.
2. **Train.cpp**: C++ script to train the camera motion classification model.
3. **Inference.cpp**: C++ script to classify camera motion in a test video using the trained model.

---

## Notes
- Ensure that the video files are in `.mp4` format.
- The Python implementation is easier to set up and run, while the C++ implementation is more performant but requires additional setup.
- For the C++ implementation, serialization of the PCA and SVM models is not fully implemented. You may need to extend the code to handle this.
