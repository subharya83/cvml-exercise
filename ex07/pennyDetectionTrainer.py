import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
import cv2

# Dataset Paths
train_images_path = "path/to/train/images"  # Directory with training images
train_annotations_path = "path/to/train/annotations"  # Directory with annotations
val_images_path = "path/to/val/images"  # Directory with validation images
val_annotations_path = "path/to/val/annotations"  # Directory with annotations

# Load Dataset
def load_dataset(image_dir, annotation_dir, img_size=(224, 224)):
    """
    Loads and preprocesses images and annotations.

    Parameters:
        image_dir (str): Directory with images.
        annotation_dir (str): Directory with annotations (e.g., Pascal VOC XML).
        img_size (tuple): Target image size.

    Returns:
        images (np.ndarray): Preprocessed images.
        labels (np.ndarray): Corresponding labels (bounding boxes and classes).
    """
    images = []
    bboxes = []
    classes = []

    for file in os.listdir(image_dir):
        if file.endswith(".jpg") or file.endswith(".png"):
            # Load and resize the image
            img = cv2.imread(os.path.join(image_dir, file))
            img = cv2.resize(img, img_size)
            images.append(img)

            # Load corresponding annotation
            annotation_file = os.path.join(annotation_dir, file.replace(".jpg", ".xml"))
            bbox, cls = parse_annotation(annotation_file)  # Implement parse_annotation for XML
            bboxes.append(bbox)
            classes.append(cls)

    images = np.array(images) / 255.0  # Normalize pixel values
    return np.array(images), {"bboxes": np.array(bboxes), "classes": np.array(classes)}

def parse_annotation(annotation_file):
    """
    Parses the annotation file (Pascal VOC format).
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(annotation_file)
    root = tree.getroot()

    bbox = []
    cls = []

    for obj in root.findall("object"):
        # Extract class label
        label = obj.find("name").text
        cls.append(1 if label == "penny" else 0)  # 1 for penny, 0 otherwise

        # Extract bounding box
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        bbox.append([xmin, ymin, xmax, ymax])

    return bbox, cls

# Model Definition
def build_model(input_shape=(224, 224, 3), num_classes=1):
    """
    Builds the CNN model for object detection.
    """
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)

    # Add custom layers
    x = base_model.output
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)

    # Output layers
    bbox_output = Dense(4, activation="linear", name="bbox_output")(x)  # For bounding box regression
    class_output = Dense(num_classes, activation="sigmoid", name="class_output")(x)  # For binary classification

    model = Model(inputs=base_model.input, outputs=[bbox_output, class_output])
    return model

# Compile the Model
model = build_model()
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss={
        "bbox_output": "mean_squared_error",
        "class_output": "binary_crossentropy",
    },
    metrics={
        "bbox_output": "mse",
        "class_output": "accuracy",
    },
)

# Load Data
train_images, train_labels = load_dataset(train_images_path, train_annotations_path)
val_images, val_labels = load_dataset(val_images_path, val_annotations_path)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

train_generator = datagen.flow(
    train_images, {"bbox_output": train_labels["bboxes"], "class_output": train_labels["classes"]}, batch_size=32
)

# Train the Model
history = model.fit(
    train_generator,
    validation_data=(
        val_images,
        {"bbox_output": val_labels["bboxes"], "class_output": val_labels["classes"]},
    ),
    epochs=20,
    batch_size=32,
)

# Save the Model
model.save("penny_detection_model.h5")

