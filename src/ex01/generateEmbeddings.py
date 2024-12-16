import os
import argparse
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
import json

class VideoEmbeddingGenerator:
    def __init__(self, model_name='resnet50'):
        """
        Initialize the embedding generator with a pre-trained model.
        
        :param model_name: Name of the model to use for feature extraction
        """
        # Load the pre-trained model
        if model_name == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif model_name == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Remove the final classification layer
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        # Set the model to evaluation mode
        self.model.eval()
        
        # Define image transformation
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
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
    embedding_generator = VideoEmbeddingGenerator()
    
    # Open video
    if os.path.exists(input_sequence):
        
    video = cv2.VideoCapture(input_video)
    
    # Prepare output data structure
    frame_index = []
    
    try:
        frame_count = 0
        while True:
            # Read frame
            ret, frame = video.read()
            
            # Break if no more frames
            if not ret:
                break
            
            # Sample frames
            if frame_count % sample_rate == 0:
                # Generate embedding
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
        video.release()
    
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
    process_video(args.input_video, args.output_index, args.sample_rate)

if __name__ == '__main__':
    main()