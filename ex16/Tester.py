import numpy as np
import joblib
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
    # Load the trained model
    classifier, pca = joblib.load('trained_model.pkl')

    # Test on a new video
    test_video_path = "path_to_test_video.mp4"
    predicted_class = test_classifier(classifier, pca, test_video_path)
    print(f"Predicted Class: {predicted_class}")

if __name__ == "__main__":
    main()