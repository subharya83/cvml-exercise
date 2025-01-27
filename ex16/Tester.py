import os
import joblib
from CamMotionClassification import CamMotionClassifier

def test_classifier(classifier, test_video_path):
    features = classifier.extract_features(test_video_path)
    if len(features) > 0:
        prediction = classifier.predict(np.mean(features, axis=0).reshape(1, -1))
        return prediction[0]
    else:
        return "Unknown"

def main():
    # Load the trained classifier
    classifier = joblib.load('trained_classifier.pkl')

    # Test on a new video
    test_video_path = "path_to_test_video"
    predicted_class = test_classifier(classifier, test_video_path)
    print(f"Predicted Class: {predicted_class}")

if __name__ == "__main__":
    main()