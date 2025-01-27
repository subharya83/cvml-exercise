import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import joblib

def load_dataset(dataset_file):
    data = np.load(dataset_file)
    X = data['X']
    y = data['y']
    return X, y

def main():
    dataset_file = "dataset.npz"
    X, y = load_dataset(dataset_file)

    # Reduce dimensionality using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Train an SVM classifier
    classifier = SVC(kernel='linear')
    classifier.fit(X_pca, y)

    # Save the trained model
    joblib.dump((classifier, pca), 'trained_model.pkl')
    print("Model trained and saved to trained_model.pkl")

if __name__ == "__main__":
    main()