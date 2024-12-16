import os
import argparse
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import json

class VideoEmbeddingGenerator:
    def __init__(self, model_name='resnet50', download_weights=True):
        """
        Initialize the embedding generator with a pre-trained model.
        
        :param model_name: Name of the model to use for feature extraction
        :param download_weights: Whether to download weights if not present
        """
        # Determine weights file path
        weights_dir = os.path.join(os.getcwd(), '../weights')
        os.makedirs(weights_dir, exist_ok=True)
        
        # Mapping of model names to their weight URLs and local filenames
        model_weights = {
            'resnet50': {
                'url': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
                'filename': 'resnet50.pth'
            },
            'resnet18': {
                'url': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
                'filename': 'resnet18.pth'
            }
        }
        
        if model_name not in model_weights:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Path for local weight file
        local_weight_path = os.path.join(weights_dir, model_weights[model_name]['filename'])
        
        # Download weights if requested and not already present
        if download_weights and not os.path.exists(local_weight_path):
            try:
                print(f"Downloading weights for {model_name}...")
                torch.hub.download_url_to_file(model_weights[model_name]['url'], local_weight_path)
                print(f"Weights downloaded to {local_weight_path}")
            except Exception as e:
                print(f"Error downloading weights: {e}")
                local_weight_path = None
        
        # Load the model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=False)
            if local_weight_path and os.path.exists(local_weight_path):
                self.model.load_state_dict(torch.load(local_weight_path))
        elif model_name == 'resnet18':
            self.model = models.resnet18(pretrained=False)
            if local_weight_path and os.path.exists(local_weight_path):
                self.model.load_state_dict(torch.load(local_weight_path))
        
        # Remove the final classification layer
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        # Set the model to evaluation mode
        self.model.eval()
        
        # Define image transformation
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def generate_embedding(self, frame):
        """
        Generate an embedding for a given frame.
        
        :param frame: OpenCV frame (numpy array)
        :return: Numpy array representing the frame embedding
        """
        # Transform the frame
        input_tensor = self.transform(frame).unsqueeze(0)
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.model(input_tensor)
        
        # Convert to numpy and flatten
        embedding_np = embedding.squeeze().numpy()
        
        return embedding_np

def process_input(input_sequence, output_index):
    """
    Process video and generate embeddings for sampled frames.
    
    :param input_sequence: Path to input video file or directory containing images
    :param output_index: Path to output index file
    """
    # Initialize embedding generator
    embedding_generator = VideoEmbeddingGenerator(model_name='resnet50', download_weights=True)
    
    seq = None
    isVideo = False
    if os.path.exists(input_sequence):
        if os.path.isfile(input_sequence):
            seq = cv2.VideoCapture(input_sequence)
            isVideo = True
        elif os.path.isdir(input_sequence):
            seq = sorted(os.listdir(input_sequence))
    
    # Prepare output data structure
    frame_index = []
    
    try:
        frame_count = 0
        while True:
            # Read frame
            if isVideo:
                ret, frame = seq.read()
                # Break if no more frames
                if not ret:
                    break
                frame = Image.fromarray(frame)
            else:
                frame = Image.open(seq[frame_count]).convert('RGB')
                
            embedding = embedding_generator.generate_embedding(frame)
            # Store frame information
            frame_info = {
                'frame_number': frame_count,
                'embedding': embedding.tolist()
            }
            frame_index.append(frame_info)
            frame_count += 1

    finally:
        # Release video capture
        if isVideo:
            seq.release()
    
    # Save index to file
    with open(output_index, 'w') as f:
        json.dump(frame_index, f)
    
    print(f"Processed {len(frame_index)} frames. Index saved to {output_index}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Generate embeddings for video frames')
    parser.add_argument('-i', help='Path to input video or directory containing images')
    parser.add_argument('-o', help='Path to output embedding index file')
   
    # Parse arguments
    args = parser.parse_args()
    
    # Process video
    process_input(args.i, args.o)

if __name__ == '__main__':
    main()