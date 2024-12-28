## Training a simple CNN to detect pennies

Code to train a Convolutional Neural Network (CNN) for detecting and classifying pennies from annotated images. 
This example uses **TensorFlow/Keras** and assumes we  have annotated images where each coin is labeled as a penny or another coin type.

---

### Steps to Train the CNN

1. **Prepare the Dataset**:
   - Dataset should have images and corresponding annotations (bounding boxes and classes).
   - Annotations can be in formats like Pascal VOC (XML) or COCO (JSON).

2. **Data Preprocessing**:
   - Load and resize the images.
   - Normalize pixel values.
   - Prepare bounding box and class labels.

3. **Model Definition**:
   - Use a pre-trained model (like MobileNet or ResNet) as the backbone and add custom layers for bounding box regression and classification.

4. **Training the Model**:
   - Train the model on the annotated dataset.
   - Use data augmentation to improve generalization.

5. **Save the Model**:
   - Save the trained model for inference.



### Explanation of the Code

1. **Dataset Loading**:
   - Images are loaded from directories.
   - Annotations are parsed (assumes Pascal VOC XML format).
   - Each coin is labeled with its bounding box and class.

2. **Model Definition**:
   - A pre-trained MobileNetV2 is used as the backbone.
   - Two outputs are added:
     - **Bounding Box Regression**: Predicts the coordinates of the bounding box.
     - **Classification**: Predicts whether the coin is a penny or not.

3. **Training**:
   - Uses `ImageDataGenerator` for augmentation.
   - Trains with both bounding box regression and classification tasks.

4. **Saving the Model**:
   - The trained model is saved for later use in inference.
