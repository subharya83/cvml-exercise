import os
import argparse
import json
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2

class ImageEmbeddingGenerator:
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
    
    def generate_embedding(self, image_path):
        """
        Generate an embedding for a given image.
        
        :param image_path: Path to the image file
        :return: Numpy array representing the image embedding
        """
        # Open the image
        image = Image.open(image_path).convert('RGB')
        
        # Transform the image
        input_tensor = self.transform(image).unsqueeze(0)
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.model(input_tensor)
        
        # Convert to numpy and flatten
        embedding_np = embedding.squeeze().numpy()
        
        return embedding_np
    
    @staticmethod
    def compute_cosine_similarity(embedding1, embedding2):
        """
        Compute cosine similarity between two embeddings.
        
        :param embedding1: First embedding
        :param embedding2: Second embedding
        :return: Cosine similarity score
        """
        # Ensure embeddings are numpy arrays
        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)
        
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        # Compute dot product
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        return similarity

def search_most_similar_frame(query_image, index_file):
    """
    Find the most similar frame from the video index.
    
    :param query_image: Path to the query image
    :param index_file: Path to the video frame embedding index
    :return: Information about the most similar frame
    """
    # Load embedding generator
    embedding_generator = ImageEmbeddingGenerator()
    
    # Generate embedding for query image
    query_embedding = embedding_generator.generate_embedding(query_image)
    
    # Load frame index
    with open(index_file, 'r') as f:
        frame_index = json.load(f)
    
    # Find most similar frame
    best_similarity = -1
    best_frame = None
    
    for frame_info in frame_index:
        # Compute similarity
        similarity = ImageEmbeddingGenerator.compute_cosine_similarity(query_embedding, frame_info['embedding'])
        
        # Update best match
        if similarity > best_similarity:
            best_similarity = similarity
            best_frame = frame_info
    
    return best_frame, best_similarity

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Search for most similar video frame')
    parser.add_argument('-q', required=True, help='Path to query image')
    parser.add_argument('-i', required=True, help='Path to video frame embedding index')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Perform search
    best_frame, similarity = search_most_similar_frame(args.q, args.i)
    
    # Print results
    print(f"Most Similar Frame:")
    print(f"Frame Number: {best_frame['frame_number']}")
    print(f"Cosine Similarity: {similarity}")

if __name__ == '__main__':
    main()