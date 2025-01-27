import os
import numpy as np
from CamMotionClassification import CamMotionClassifier

def create_dataset(video_dir, output_file):
    classifier = CamMotionClassifier()
    X = []
    y = []

    # Iterate through all video files in the directory
    for video_file in os.listdir(video_dir):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(video_dir, video_file)
            shot_name = video_file.split('_')[0]  # Extract shot name from filename

            # Extract features from the video
            features = classifier.extract_features(video_path)
            if len(features) > 0:
                X.append(np.mean(features, axis=0))  # Use mean feature vector
                y.append(shot_name)

    # Save the dataset to a file
    np.savez(output_file, X=np.array(X), y=np.array(y))
    print(f"Dataset created and saved to {output_file}")

if __name__ == "__main__":
    video_dir = "path_to_video_shots"  # Directory containing video shots
    output_file = "dataset.npz"  # Output file for the dataset
    create_dataset(video_dir, output_file)