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
    parser.add_argument("-v", type=str, required=True, help="Directory containing video shots.")
    parser.add_argument("-d", type=str, default="output/dataset.npz", help="Output file for the dataset.")
    parser.add_argument("-m", type=str, default="output/CamMotionModel.pkl", help="Output file for the trained model.")
    args = parser.parse_args()

    # Create dataset if it doesn't exist
    if not os.path.exists(args.d):
        print(f"Dataset file {args.d} not found. Creating dataset...")
        classifier = CamMotionClassifier()
        classifier.create_dataset(args.v, args.d)

    # Load the dataset
    X, y = load_dataset(args.d)

    # Reduce dimensionality using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Train an SVM classifier
    classifier = SVC(kernel='linear')
    classifier.fit(X_pca, y)

    # Save the trained model
    joblib.dump((classifier, pca), args.m)
    print(f"Model trained and saved to {args.m}")

if __name__ == "__main__":
    main()