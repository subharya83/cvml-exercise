# Shot Classification with SGNet

This project implements the SGNet model for classifying movie shots based on scale and movement. The model is trained on a dataset of movie shots and can be used to classify new images or videos.

## Requirements

- Python 3.6+
- PyTorch 1.0+
- torchvision
- OpenCV
- pandas
- PIL (Pillow)

## Installation


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

The SGNet model consists of:
- A `SubjectMapGenerator` for generating subject maps.
- A `VarianceMapModule` for generating variance maps.
- A ResNet50 backbone for feature extraction.
- Fully connected layers for scale and movement classification.
