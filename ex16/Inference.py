import numpy as np
import joblib
import argparse
from CamMotionClassification import CamMotionClassifier

def test_classifier(classifier, pca, test_video_path):
    # Extract features from the test video
    classifier = CamMotionClassifier()
    features = classifier.extract_features(test_video_path)
    if len(features) > 0:
        # Reduce dimensionality using PCA
        features_pca = pca.transform(np.mean(features, axis=0).reshape(1, -1))
        prediction = classifier.predict(features_pca)
        return prediction[0]
    else:
        return "Unknown"

def main():
    parser = argparse.ArgumentParser(description="Test a camera motion classifier.")
    parser.add_argument("-m", type=str, default="output/CamMotionModel.pkl", help="File containing the trained model.")
    parser.add_argument("-t", type=str, required=True, help="Path to the test video.")
    args = parser.parse_args()

    # Load the trained model
    classifier, pca = joblib.load(args.m)

    # Test on the provided video
    predicted_class = test_classifier(classifier, pca, args.t)
    print(f"Predicted Class: {predicted_class}")

if __name__ == "__main__":
    main()
