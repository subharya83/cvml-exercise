import os
import numpy as np
from CamMotionClassification import CamMotionClassifier

def load_dataset(dataset_path):
    X = []
    y = []
    class_labels = os.listdir(dataset_path)
    for label in class_labels:
        class_path = os.path.join(dataset_path, label)
        for video_file in os.listdir(class_path):
            video_path = os.path.join(class_path, video_file)
            classifier = CamMotionClassifier()
            features = classifier.extract_features(video_path)
            if len(features) > 0:
                X.append(np.mean(features, axis=0))  # Use mean feature vector
                y.append(label)
    return np.array(X), np.array(y)

def main():
    dataset_path = "path_to_cinematographic_shot_dataset"
    X, y = load_dataset(dataset_path)

    classifier = CamMotionClassifier()
    classifier.train(X, y)

    # Save the trained model (optional)
    import joblib
    joblib.dump(classifier, 'trained_classifier.pkl')

if __name__ == "__main__":
    main()