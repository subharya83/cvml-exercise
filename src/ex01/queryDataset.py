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
        similarity = ImageEmbeddingGenerator.compute_cosine_similarity(
            query_embedding, 
            frame_info['embedding']
        )
        
        # Update best match
        if similarity > best_similarity:
            best_similarity = similarity
            best_frame = frame_info
    
    return best_frame, best_similarity

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Search for most similar video frame')
    parser.add_argument('query_image', help='Path to query image')
    parser.add_argument('index_file', help='Path to video frame embedding index')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Perform search
    best_frame, similarity = search_most_similar_frame(
        args.query_image, 
        args.index_file
    )
    
    # Print results
    print(f"Most Similar Frame:")
    print(f"Frame Number: {best_frame['frame_number']}")
    print(f"Cosine Similarity: {similarity}")

if __name__ == '__main__':
    main()