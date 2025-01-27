import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import joblib
import argparse
from CamMotionClassification import CamMotionClassifier

def load_dataset(dataset_file):
    data = np.load(dataset_file)
    X = data['X']
    y = data['y']
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Train a camera motion classifier.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video shots.")
    parser.add_argument("--dataset_file", type=str, default="dataset.npz", help="Output file for the dataset.")
    parser.add_argument("--model_file", type=str, default="trained_model.pkl", help="Output file for the trained model.")
    args = parser.parse_args()

    # Create dataset if it doesn't exist
    if not os.path.exists(args.dataset_file):
        print(f"Dataset file {args.dataset_file} not found. Creating dataset...")
        classifier = CamMotionClassifier()
        classifier.create_dataset(args.video_dir, args.dataset_file)

    # Load the dataset
    X, y = load_dataset(args.dataset_file)

    # Reduce dimensionality using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Train an SVM classifier
    classifier = SVC(kernel='linear')
    classifier.fit(X_pca, y)

    # Save the trained model
    joblib.dump((classifier, pca), args.model_file)
    print(f"Model trained and saved to {args.model_file}")

if __name__ == "__main__":
    main()