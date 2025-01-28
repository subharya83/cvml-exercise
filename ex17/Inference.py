import torch
from SubCentShotClassification import SGNet
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import cv2
import argparse
import numpy as np

def preprocess_image(image):
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def process_image(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image = preprocess_image(image)

    with torch.no_grad():
        scale_logits, movement_logits = model(image)
        scale_probs = torch.softmax(scale_logits, dim=1).squeeze().numpy()
        movement_probs = torch.softmax(movement_logits, dim=1).squeeze().numpy()

    return scale_probs, movement_probs

def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    scale_probs_list = []
    movement_probs_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image = preprocess_image(image)

        with torch.no_grad():
            scale_logits, movement_logits = model(image)
            scale_probs = torch.softmax(scale_logits, dim=1).squeeze().numpy()
            movement_probs = torch.softmax(movement_logits, dim=1).squeeze().numpy()

        scale_probs_list.append(scale_probs)
        movement_probs_list.append(movement_probs)

    cap.release()

    # Average probabilities across all frames
    avg_scale_probs = np.mean(scale_probs_list, axis=0)
    avg_movement_probs = np.mean(movement_probs_list, axis=0)

    return avg_scale_probs, avg_movement_probs

def main():
    parser = argparse.ArgumentParser(description="Perform inference on an image or video.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the input image or video.")
    args = parser.parse_args()

    # Load the trained model
    model = SGNet()
    model.load_state_dict(torch.load('sgnet_model.pth'))
    model.eval()

    # Determine if the input is an image or video
    if args.input.endswith(('.jpg', '.jpeg', '.png')):
        scale_probs, movement_probs = process_image(args.input, model)
    elif args.input.endswith(('.mp4', '.avi', '.mov')):
        scale_probs, movement_probs = process_video(args.input, model)
    else:
        raise ValueError("Unsupported file format. Please provide an image (jpg, jpeg, png) or video (mp4, avi, mov).")

    # Map predictions to class labels
    scale_classes = ['Long Shot', 'Full Shot', 'Medium Shot', 'Close-up Shot', 'Extreme Close-up Shot']
    movement_classes = ['Static Shot', 'Motion Shot', 'Push Shot', 'Pull Shot']

    # Print probabilities for each class
    print("Scale Class Probabilities:")
    for i, prob in enumerate(scale_probs):
        print(f"{scale_classes[i]}: {prob:.4f}")

    print("\nMovement Class Probabilities:")
    for i, prob in enumerate(movement_probs):
        print(f"{movement_classes[i]}: {prob:.4f}")

if __name__ == "__main__":
    main()