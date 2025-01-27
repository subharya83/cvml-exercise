import torch
from SubCentShotClassification import SGNet
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image

# Load the trained model
model = SGNet()
model.load_state_dict(torch.load('sgnet_model.pth'))
model.eval()

# Define preprocessing pipeline
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
])

# Load and preprocess the image
image_path = 'path_to_test_image.jpg'
image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0)

# Perform inference
with torch.no_grad():
    scale_logits, movement_logits = model(image)
    scale_pred = torch.argmax(scale_logits, dim=1).item()
    movement_pred = torch.argmax(movement_logits, dim=1).item()

# Map predictions to class labels
scale_classes = ['Long Shot', 'Full Shot', 'Medium Shot', 'Close-up Shot', 'Extreme Close-up Shot']
movement_classes = ['Static Shot', 'Motion Shot', 'Push Shot', 'Pull Shot']

print(f'Predicted Scale: {scale_classes[scale_pred]}')
print(f'Predicted Movement: {movement_classes[movement_pred]}')