### Updated README.md

```markdown
# Shot Classification with SGNet: A Deep Dive into Movie Shot Analysis

Welcome to the **Shot Classification with SGNet** project! This project is designed to classify movie shots based on their scale and movement using the **SGNet** model. Whether you're a computer vision enthusiast or a film buff, this project offers a fascinating intersection of deep learning and cinematic analysis.

## Why This Project?

- **Cinematic Insight**: Understand how different shot scales and movements contribute to storytelling in movies.
- **Hands-On Learning**: Get hands-on experience with state-of-the-art deep learning models in computer vision.
- **Real-World Application**: Apply your skills to classify shots in real movies or even your own videos.

## Requirements

To get started, you'll need the following:

- Python 3.6+
- PyTorch 1.0+
- torchvision
- OpenCV
- pandas
- PIL (Pillow)

You can install all the required packages using the `requirements.txt` file provided in this repository.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/shot-classification-sgnet.git
   cd shot-classification-sgnet
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Build and run with Docker** (optional):
   ```bash
   docker build -t sgnet .
   docker run -it sgnet
   ```

## Usage

### Building the Dataset

To build the dataset from a directory of images/videos and a CSV file, run:
```bash
python Train.py -d dataset.pth -r /path/to/root -c /path/to/csv
```

### Training the Model

To train the model using a pre-built dataset, run:
```bash
python Train.py -t dataset.pth -o model.pth
```

### Inference

To perform inference on an image or video, run:
```bash
python Inference.py -i /path/to/input
```

## File Structure

- `Train.py`: Script for building the dataset and training the model.
- `Inference.py`: Script for performing inference on images or videos.
- `SubCentShotClassification.py`: Contains the SGNet model and related modules.

## Model Architecture

The SGNet model is a powerful deep learning architecture designed specifically for shot classification. It consists of:

- **SubjectMapGenerator**: Generates subject maps to focus on the main subjects in the shot.
- **VarianceMapModule**: Captures the variance in movement across the shot.
- **ResNet50 Backbone**: Extracts high-level features from the input images.
- **Fully Connected Layers**: Classifies the shot based on scale and movement.

## Docker Support

We've included a `Dockerfile` to make it easy to set up the environment and run the project in a containerized environment. This is especially useful for ensuring consistency across different systems.

## Contributing

We welcome contributions! If you have any ideas, bug fixes, or improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```

### requirements.txt

```plaintext
torch>=1.0.0
torchvision>=0.2.1
opencv-python>=4.0.0
pandas>=1.0.0
Pillow>=6.0.0
numpy>=1.18.0
```