import torch
from SubCentShotClassification import SGNet
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import cv2

def preprocess_image(image):
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    scale_predictions = []
    movement_predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image = preprocess_image(image)

        with torch.no_grad():
            scale_logits, movement_logits = model(image)
            scale_pred = torch.argmax(scale_logits, dim=1).item()
            movement_pred = torch.argmax(movement_logits, dim=1).item()

        scale_predictions.append(scale_pred)
        movement_predictions.append(movement_pred)

    cap.release()
    return scale_predictions, movement_predictions

def main():
    # Load the trained model
    model = SGNet()
    model.load_state_dict(torch.load('sgnet_model.pth'))
    model.eval()

    # Process video
    video_path = 'path_to_test_video.mp4'
    scale_predictions, movement_predictions = process_video(video_path, model)

    # Map predictions to class labels
    scale_classes = ['Long Shot', 'Full Shot', 'Medium Shot', 'Close-up Shot', 'Extreme Close-up Shot']
    movement_classes = ['Static Shot', 'Motion Shot', 'Push Shot', 'Pull Shot']

    # Aggregate predictions (e.g., majority vote)
    final_scale_pred = max(set(scale_predictions), key=scale_predictions.count)
    final_movement_pred = max(set(movement_predictions), key=movement_predictions.count)

    print(f'Predicted Scale: {scale_classes[final_scale_pred]}')
    print(f'Predicted Movement: {movement_classes[final_movement_pred]}')

if __name__ == "__main__":
    main()