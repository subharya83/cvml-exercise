import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

class Embeddings:
    def __init__(self, model_name='resnet50', download_weights=True):
        """
        Initialize the embedding generator with a pre-trained model.
        
        :param model_name: Name of the model to use for feature extraction
        :param download_weights: Whether to download weights if not present
        """
        # Determine weights file path
        weights_dir = os.path.join(os.getcwd(), 'weights')
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
            self.model = models.resnet50(weights=None)
            if local_weight_path and os.path.exists(local_weight_path):
                self.model.load_state_dict(torch.load(local_weight_path))
        elif model_name == 'resnet18':
            self.model = models.resnet18(weights=None)
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
    
    def generate_embedding(self, img=None):
        """
        Generate an embedding for a given image.
        
        :param img: image object Pillow
        :return: Numpy array representing the image embedding
        """
        
        # Transform the image
        input_tensor = self.transform(img).unsqueeze(0)
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.model(input_tensor)
        
        # Convert to numpy and flatten
        embedding_np = embedding.squeeze().numpy()
        
        return embedding_np
    
    def compute_similarity(self, embedding1, embedding2):
        """
        Compute cosine similarity between two embeddings.
        
        :param embedding1: First embedding
        :param embedding2: Second embedding
        :return: Cosine similarity score
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        # Compute dot product
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        return similarity
