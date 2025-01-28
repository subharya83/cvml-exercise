import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from SubCentShotClassification import SGNet
from torch.nn import CrossEntropyLoss
import argparse
import os
import cv2
import pandas as pd
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import numpy as np

def get_file_type(file_path):
    # Check magic numbers to determine file type
    with open(file_path, 'rb') as f:
        header = f.read(12)
    
    # Check for common image formats
    if len(header) >= 2 and header.startswith(b'\xFF\xD8'):
        return 'image'  # JPEG
    if len(header) >= 8 and header.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image'  # PNG
    if len(header) >= 6 and (header.startswith(b'GIF87a') or header.startswith(b'GIF89a')):
        return 'image'  # GIF
    # Check for video formats
    if len(header) >= 8:
        if header[4:8] == b'ftyp':  # MP4, MOV
            return 'video'
        if header.startswith(b'RIFF') and len(header) >= 12 and header[8:12] == b'AVI ':
            return 'video'  # AVI
    # Fallback to file extension if magic number not recognized
    ext = os.path.splitext(file_path)[1].lower()
    image_exts = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']
    if ext in image_exts:
        return 'image'
    elif ext in video_exts:
        return 'video'
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

class MovieShotsDataset(Dataset):
    def __init__(self, root, csv_file, transform=None):
        self.root = root
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root, self.data.iloc[idx, 0])
        scale_label = self.data.iloc[idx, 1]
        movement_label = self.data.iloc[idx, 2]

        file_type = get_file_type(file_path)

        if file_type == 'image':
            # Load image directly
            image = Image.open(file_path)
        elif file_type == 'video':
            # Extract random frame from video
            cap = cv2.VideoCapture(file_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0:
                raise ValueError(f"Video {file_path} has no frames.")
            frame_idx = np.random.randint(0, frame_count)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise ValueError(f"Unable to read frame {frame_idx} from video {file_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            raise ValueError(f"Unsupported file type for {file_path}")

        if self.transform:
            image = self.transform(image)

        return image, scale_label, movement_label

def build_dataset(root, csv_file, output_file):
    dataset = MovieShotsDataset(root=root, csv_file=csv_file, transform=Compose([Resize((224, 224)), ToTensor()]))
    torch.save(dataset, output_file)
    print(f"Dataset built and saved to {output_file}")

def train_model(dataset_file, output_model_file):
    # Load dataset
    dataset = torch.load(dataset_file)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Initialize model, loss, and optimizer
    model = SGNet()
    criterion_scale = CrossEntropyLoss()
    criterion_movement = CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    num_epochs = 60
    for epoch in range(num_epochs):
        model.train()
        for i, (images, scale_labels, movement_labels) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            scale_logits, movement_logits = model(images)

            # Compute loss
            loss_scale = criterion_scale(scale_logits, scale_labels)
            loss_movement = criterion_movement(movement_logits, movement_labels)
            total_loss = loss_scale + loss_movement

            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {total_loss.item():.4f}')

        # Adjust learning rate
        if epoch == 20 or epoch == 40:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10

    # Save the model
    torch.save(model.state_dict(), output_model_file)
    print(f"Model saved to {output_model_file}")

def main():
    parser = argparse.ArgumentParser(description="Train the SGNet model for shot classification.")
    parser.add_argument('-d', '--dataset', type=str, help="Build dataset from files and CSV. Usage: -d dataset.ext")
    parser.add_argument('-t', '--train', type=str, help="Train the model using a pre-built dataset. Usage: -t dataset.ext")
    parser.add_argument('-o', '--output', type=str, help="Output model file. Usage: -o model.pth")
    parser.add_argument('-r', '--root', type=str, help="Root directory containing files.")
    parser.add_argument('-c', '--csv', type=str, help="CSV file containing filenames and labels.")
    args = parser.parse_args()

    if args.dataset:
        if not args.root or not args.csv:
            raise ValueError("Both --root and --csv arguments are required to build the dataset.")
        build_dataset(args.root, args.csv, args.dataset)
    elif args.train:
        if not args.output:
            raise ValueError("The --output argument is required to save the trained model.")
        train_model(args.train, args.output)
    else:
        raise ValueError("Either --dataset or --train argument must be provided.")

if __name__ == "__main__":
    main()