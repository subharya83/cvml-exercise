# Dimension Estimation Using YOLOv5 and Reference Objects

A Python-based tool for estimating the dimensions of objects (like trees or buildings) in an image using YOLOv5 and a reference object (a standard soda can). This project is designed to be a practical example of how computer vision techniques can be applied to real-world problems, such as estimating the size of objects in images where no direct scale is available.

## 🚀 Overview

The project leverages the power of **YOLOv5**, a state-of-the-art object detection model, to detect objects in an image. By using a known reference object (a soda can with standard dimensions), the tool estimates the height and width of other objects in the scene. This is particularly useful in scenarios where you need to measure objects in images without access to a physical scale.

### Key Features:
- **YOLOv5 Integration**: The project uses YOLOv5 for object detection, providing fast and accurate detection of objects in images.
- **Reference Object Scaling**: By detecting a soda can (or any known object), the tool calculates the scale of the image and estimates the dimensions of other objects.
- **Custom Object Detection**: If YOLOv5 fails to detect the target object (e.g., a tree or building), the tool uses basic computer vision techniques (color segmentation and edge detection) to estimate the object's bounding box.
- **Visualization**: The tool generates an output image with bounding boxes and estimated dimensions overlaid on the original image.

## 📂 Repository Structure

- **`download.sh`**: A Bash script to download the YOLOv5 repository and pre-trained weights.
- **`estimateDims.py`**: The main Python script that performs object detection, dimension estimation, and visualization.
- **`README.md`**: This file, providing an overview of the project and instructions for use.

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch (with CUDA if available)
- OpenCV
- Other dependencies listed in `yolov5/requirements.txt`

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/dimension-estimation.git
   cd dimension-estimation
   ```

2. **Run the Setup Script**:
   The `download.sh` script will download the YOLOv5 repository and the necessary weights.
   ```bash
   chmod +x download.sh
   ./download.sh
   ```

3. **Install Python Dependencies**:
   Install the required Python packages using pip.
   ```bash
   pip install -r yolov5/requirements.txt
   ```

## 🖼️ Usage

To estimate the dimensions of an object in an image, run the `estimateDims.py` script with the following command:

```bash
python estimateDims.py -i path/to/your/image.jpg -o output_image.jpg -t tree
```

### Arguments:
- `-i` or `--input`: Path to the input image.
- `-o` or `--output`: (Optional) Path to save the output image with visualization. If not provided, the output will be saved with a `_result` suffix.
- `-t` or `--type`: (Optional) Type of object to measure (`tree` or `building`). Default is `tree`.

### Example Output:
```bash
Processing image: example.jpg
Target type: tree

Estimated dimensions:
Height: 12.34 meters
Width: 4.56 meters
Reference object: soda can (height: 12.2 cm)

Visualization saved to: example_result.jpg
```
### Prerequisites for C++ Version:

- **OpenCV**: Install OpenCV with DNN module support.
- **YOLOv5 ONNX Model**: Convert the YOLOv5 model to ONNX format and place it in the `weights` directory.
- **C++ Compiler**: Use a C++ compiler that supports C++17 (e.g., GCC, Clang, or MSVC).

### Compilation and Execution:

1. **Compile the C++ Code**:
   ```bash
   g++ -std=c++17 -o estimateDims estimateDims.cpp `pkg-config --cflags --libs opencv4`
   ```

2. **Run the Executable**:
   ```bash
   ./estimateDims input_image.jpg output_image.jpg tree
   ```

### Computational Workflow
```
+-------------------+       +-------------------+       +-------------------+
|  Command Line     |       |  Dimension        |       |  YOLOv5 Setup     |
|  Arguments        | ----> |  Estimator        | <->   | (download_yolo.sh)|
|  (parse_arguments)|       |  Initialization   |       |                   |
+-------------------+       +-------------------+       +-------------------+
        |                                                   |
        V                                                   V
+-------------------+       +-------------------+       +-------------------+
|  Image Preprocess |       |  Object Detection |       |  YOLOv5 Model     |
|  (preprocess_image)| <->  |  (detect_objects) | <---  |  (attempt_load)   |
+-------------------+       +-------------------+       +-------------------+
                                |                           
                                v                           
+-------------------+       +-------------------+       +----------------------+
|  Dimension        |       |  Tree/Building    |       |  Visualization       |
|  Estimation       | <---- |  Detection        | --->  |  (_visualize_results)|
+-------------------+       +-------------------+       +----------------------+
       
```


### Notes:
- The C++ version assumes that the YOLOv5 model is already converted to ONNX format. You can convert the PyTorch model to ONNX using the `torch.onnx.export` function in Python.
- The C++ version is more performant than the Python version, especially for real-time applications, but requires more setup and configuration.

This C++ implementation provides a solid foundation for further optimization and extension, making it suitable for graduate students interested in high-performance computer vision applications.


## 🧠 How It Works

1. **Object Detection**: The YOLOv5 model detects objects in the image, including the reference object (soda can) and the target object (tree or building).
2. **Bounding Box Extraction**: The bounding boxes for the reference and target objects are extracted.
3. **Scaling**: Using the known dimensions of the soda can, the tool calculates the scale of the image and estimates the dimensions of the target object.
4. **Visualization**: The tool draws bounding boxes and estimated dimensions on the image and saves the result.

### Fallback Mechanisms:
- If YOLOv5 fails to detect the target object, the tool uses basic computer vision techniques:
  - **Tree Detection**: Color segmentation is used to detect green areas (trees).
  - **Building Detection**: Edge detection is used to detect rectangular shapes (buildings).

