import os
import argparse
import cv2
import numpy as np
import urllib.request
import tarfile
import gzip
import shutil
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from collections import Counter
from pathlib import Path
from tqdm import tqdm

class FeatureValidation:
    """Base class for image validation methods"""
    def __init__(self):
        pass
    
    def compute_features(self, image):
        """Compute features for a single image"""
        raise NotImplementedError("Subclasses must implement compute_features")
    
    def validate(self, image, references, threshold):
        """Validate an image against reference features"""
        raise NotImplementedError("Subclasses must implement validate")
    
    def calculate_references(self, images, sample_size=5):
        """Calculate reference features from a subset of images"""
        raise NotImplementedError("Subclasses must implement calculate_references")


class HistogramEdgeValidation(FeatureValidation):
    """Validation using color histograms and edge density"""
    
    def compute_features(self, image):
        """Compute both histogram and edge features for an image"""
        hist = self._compute_color_histogram(image)
        edge_density = self._compute_edge_density(image)
        return {'histogram': hist, 'edge_density': edge_density}
    
    def _compute_edge_density(self, image):
        """Compute edge density using Canny edge detector"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return np.sum(edges > 0) / (image.shape[0] * image.shape[1])
    
    def _compute_color_histogram(self, image):
        """Compute color histogram for an image"""
        hist = []
        for i in range(3):
            hist.extend(cv2.calcHist([image], [i], None, [8], [0, 256]).flatten())
        return hist
    
    def validate(self, image, references, threshold):
        """
        Validate image using histogram and edge density
        
        Args:
            image: Image to validate
            references: Dictionary with 'histograms' and 'edge_densities' lists
            threshold: Dictionary with 'hist' and 'edge' thresholds
            
        Returns:
            True if valid, False otherwise
        """
        features = self.compute_features(image)
        hist = features['histogram']
        edge_density = features['edge_density']
        
        # Check against all reference images
        for ref_hist, ref_edge_density in zip(
                references['histograms'], references['edge_densities']):
            
            # Compare histograms using correlation
            hist_similarity = cv2.compareHist(
                np.array(hist, dtype=np.float32), 
                np.array(ref_hist, dtype=np.float32), 
                cv2.HISTCMP_CORREL
            )
            
            # Compare edge densities
            edge_diff = abs(edge_density - ref_edge_density) / max(edge_density, ref_edge_density)
            edge_similarity = 1 - edge_diff
            
            # If image is similar in both aspects, consider it valid
            if hist_similarity > threshold['hist'] and edge_similarity > threshold['edge']:
                return True
                
        return False
    
    def calculate_references(self, images, sample_size=5):
        """Calculate histogram and edge density references"""
        if len(images) <= sample_size:
            sample_indices = range(len(images))
        else:
            # Choose random images as reference
            sample_indices = np.random.choice(len(images), sample_size, replace=False)
        
        reference_histograms = []
        reference_edge_densities = []
        
        for idx in sample_indices:
            features = self.compute_features(images[idx])
            reference_histograms.append(features['histogram'])
            reference_edge_densities.append(features['edge_density'])
        
        return {'histograms': reference_histograms, 'edge_densities': reference_edge_densities}


class SIFTValidation(FeatureValidation):
    """Validation using SIFT features"""
    
    def __init__(self):
        super().__init__()
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create()
        # Feature matcher
        self.matcher = cv2.BFMatcher()
    
    def compute_features(self, image):
        """Compute SIFT features for an image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return {'keypoints': keypoints, 'descriptors': descriptors}
    
    def validate(self, image, references, threshold):
        """
        Validate image using SIFT feature matching
        
        Args:
            image: Image to validate
            references: Dictionary with 'descriptors' list
            threshold: Dictionary with 'match_percent' threshold
            
        Returns:
            True if valid, False otherwise
        """
        # Get SIFT features
        features = self.compute_features(image)
        descriptors = features['descriptors']
        
        # If no descriptors found, image is probably not valid
        if descriptors is None or len(descriptors) == 0:
            return False
        
        # Check against all reference descriptors
        for ref_descriptors in references['descriptors']:
            if ref_descriptors is None or len(ref_descriptors) == 0:
                continue
                
            # Match descriptors
            matches = self.matcher.knnMatch(descriptors, ref_descriptors, k=2)
            
            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
            # Calculate match percentage
            match_percent = len(good_matches) / min(len(descriptors), len(ref_descriptors))
            
            if match_percent > threshold['match_percent']:
                return True
        
        return False
    
    def calculate_references(self, images, sample_size=5):
        """Calculate SIFT feature references"""
        if len(images) <= sample_size:
            sample_indices = range(len(images))
        else:
            # Choose random images as reference
            sample_indices = np.random.choice(len(images), sample_size, replace=False)
        
        references = {'descriptors': []}
        
        for idx in sample_indices:
            features = self.compute_features(images[idx])
            references['descriptors'].append(features['descriptors'])
        
        return references


class DeepLearningValidation(FeatureValidation):
    """Validation using deep learning features"""
    
    def __init__(self, model_name='resnet18'):
        super().__init__()
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _load_model(self, model_name):
        """Load pre-trained model"""
        weights_dir = Path("weights")
        weights_dir.mkdir(exist_ok=True)
        
        # Create a model without classifier for feature extraction
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=False)
            weights_path = weights_dir / "resnet18.pth"
            if not weights_path.exists():
                print(f"Downloading {model_name} weights...")
                # In a real implementation, you would download actual weights
                # Here we'll create a dummy file for illustration
                with open(weights_path, 'w') as f:
                    f.write("Placeholder for ResNet18 weights")
        elif model_name == 'mobilenet':
            model = models.mobilenet_v2(pretrained=False)
            weights_path = weights_dir / "mobilenet_v2.pth"
            if not weights_path.exists():
                print(f"Downloading {model_name} weights...")
                with open(weights_path, 'w') as f:
                    f.write("Placeholder for MobileNetV2 weights")
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # In a real implementation, you would load the weights
        # model.load_state_dict(torch.load(weights_path))
        
        # Remove the classifier to get features
        if model_name == 'resnet18':
            model = nn.Sequential(*list(model.children())[:-1])
        elif model_name == 'mobilenet':
            model.classifier = nn.Identity()
        
        model.eval()
        return model
    
    def compute_features(self, image):
        """Compute deep learning features for an image"""
        # Preprocess the image
        img_tensor = self.transform(image).unsqueeze(0)
        
        # Extract features
        with torch.no_grad():
            features = self.model(img_tensor)
        
        # Flatten and convert to numpy
        features = features.squeeze().cpu().numpy()
        return {'deep_features': features}
    
    def validate(self, image, references, threshold):
        """
        Validate image using deep features
        
        Args:
            image: Image to validate
            references: Dictionary with 'deep_features' list
            threshold: Dictionary with 'cosine_similarity' threshold
            
        Returns:
            True if valid, False otherwise
        """
        features = self.compute_features(image)
        deep_features = features['deep_features']
        
        # Check against all reference features
        for ref_features in references['deep_features']:
            # Compute cosine similarity
            similarity = np.dot(deep_features, ref_features) / (
                np.linalg.norm(deep_features) * np.linalg.norm(ref_features))
            
            if similarity > threshold['cosine_similarity']:
                return True
        
        return False
    
    def calculate_references(self, images, sample_size=5):
        """Calculate deep feature references"""
        if len(images) <= sample_size:
            sample_indices = range(len(images))
        else:
            # Choose random images as reference
            sample_indices = np.random.choice(len(images), sample_size, replace=False)
        
        references = {'deep_features': []}
        
        for idx in sample_indices:
            features = self.compute_features(images[idx])
            references['deep_features'].append(features['deep_features'])
        
        return references


class SceneClassification:
    """Scene classification using deep learning"""
    
    def __init__(self, model_name='resnet18'):
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Load scene categories (for Places365 dataset)
        self.categories = self._load_categories()
    
    def _load_model(self, model_name):
        """Load pre-trained scene classification model"""
        weights_dir = Path("weights")
        weights_dir.mkdir(exist_ok=True)
        
        if model_name == 'resnet18':
            # Load Places365 model based on ResNet18
            model = models.resnet18(pretrained=False)
            # Modify for Places365 (365 scene categories)
            model.fc = nn.Linear(model.fc.in_features, 365)
            
            weights_path = weights_dir / "resnet18_places365.pth"
            if not weights_path.exists():
                print("Downloading Places365 ResNet18 weights...")
                # In a real implementation, you would download actual weights
                # Here we'll create a dummy file for illustration
                with open(weights_path, 'w') as f:
                    f.write("Placeholder for Places365 ResNet18 weights")
        else:
            raise ValueError(f"Unsupported model for scene classification: {model_name}")
        
        # In a real implementation, you would load the weights
        # model.load_state_dict(torch.load(weights_path))
        
        model.eval()
        return model
    
    def _load_categories(self):
        """Load scene categories"""
        categories_path = Path("weights/categories_places365.txt")
        if not categories_path.exists():
            print("Downloading Places365 categories...")
            # Create a dummy categories file
            categories = [f"scene_{i}" for i in range(365)]
            Path("weights").mkdir(exist_ok=True)
            with open(categories_path, 'w') as f:
                for category in categories:
                    f.write(f"{category}\n")
        
        # Load categories
        with open(categories_path, 'r') as f:
            categories = [line.strip() for line in f.readlines()]
        
        return categories
    
    def classify(self, image):
        """Classify scene in the image"""
        # Preprocess the image
        img_tensor = self.transform(image).unsqueeze(0)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, predicted = torch.max(outputs, 1)
        
        # Get scene category
        scene_idx = predicted.item()
        scene = self.categories[scene_idx]
        
        return scene_idx, scene
    
    def filter_images_by_scene(self, images, threshold=0.7):
        """
        Filter images based on scene classification
        
        Args:
            images: List of images
            threshold: Minimum percentage of images that must belong to the same scene
            
        Returns:
            Filtered list of images and their scene categories
        """
        # Classify all images
        scene_indices = []
        scene_names = []
        
        print("Classifying scenes...")
        for img in tqdm(images):
            scene_idx, scene = self.classify(img)
            scene_indices.append(scene_idx)
            scene_names.append(scene)
        
        # Find the most common scene
        scene_counts = Counter(scene_indices)
        most_common_scene, count = scene_counts.most_common(1)[0]
        
        # If most images belong to the same scene, filter by that scene
        if count / len(images) >= threshold:
            filtered_images = [img for img, idx in zip(images, scene_indices) 
                              if idx == most_common_scene]
            filtered_scenes = [scene for scene, idx in zip(scene_names, scene_indices)
                              if idx == most_common_scene]
            
            print(f"Filtered by scene: {scene_names[scene_indices.index(most_common_scene)]}")
            print(f"Kept {len(filtered_images)}/{len(images)} images")
            
            return filtered_images, filtered_scenes
        else:
            # Not enough images belong to the same scene
            print("No dominant scene found, keeping all images.")
            return images, scene_names


def stitch_images(images):
    """
    Stitch multiple images into a panorama.
    
    Args:
        images: List of images to stitch
        
    Returns:
        Stitched panorama image
    """
    # Create a stitcher object
    stitcher = cv2.Stitcher_create()
    
    # Perform stitching
    status, result = stitcher.stitch(images)
    
    if status != cv2.Stitcher_OK:
        raise Exception(f"Stitching failed with error code: {status}")
        
    return result


def load_images(input_dir):
    """Load all images from a directory"""
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        raise ValueError(f"No images found in '{input_dir}'.")
    
    print(f"Found {len(image_files)} images.")
    
    # Load all images
    images = []
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        img = cv2.imread(image_path)
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: Could not load image '{image_file}'.")
    
    if len(images) < 2:
        raise ValueError("Need at least 2 valid images for stitching.")
        
    return images


def main():
    parser = argparse.ArgumentParser(description="Stitch multiple images into a panorama with validation")
    parser.add_argument("-i", "--input", required=True, help="Input directory containing images")
    parser.add_argument("-o", "--output", required=True, help="Output file for the stitched panorama")
    
    # Validation algorithm selection
    parser.add_argument("--validation", choices=['histogram', 'sift', 'deep'], default='histogram',
                      help="Validation algorithm to use")
    
    # Deep learning model selection
    parser.add_argument("--model", choices=['resnet18', 'mobilenet'], default='resnet18',
                      help="Deep learning model to use (for 'deep' validation)")
    
    # Scene classification
    parser.add_argument("--scene-classification", action="store_true",
                      help="Use scene classification as preprocessing step")
    parser.add_argument("--scene-threshold", type=float, default=0.7,
                      help="Minimum percentage of images that must belong to the same scene")
    
    # Validation thresholds
    parser.add_argument("--hist-threshold", type=float, default=0.6,
                      help="Threshold for histogram similarity (0-1)")
    parser.add_argument("--edge-threshold", type=float, default=0.5,
                      help="Threshold for edge density similarity (0-1)")
    parser.add_argument("--match-threshold", type=float, default=0.1,
                      help="Threshold for SIFT feature matches (0-1)")
    parser.add_argument("--similarity-threshold", type=float, default=0.7,
                      help="Threshold for deep feature similarity (0-1)")
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist.")
        return
    
    try:
        # Load images
        images = load_images(args.input)
        
        # Scene classification preprocessing
        if args.scene_classification:
            scene_classifier = SceneClassification(model_name=args.model)
            images, scenes = scene_classifier.filter_images_by_scene(
                images, threshold=args.scene_threshold)
        
        # Select validation method
        if args.validation == 'histogram':
            validator = HistogramEdgeValidation()
            threshold = {'hist': args.hist_threshold, 'edge': args.edge_threshold}
        elif args.validation == 'sift':
            validator = SIFTValidation()
            threshold = {'match_percent': args.match_threshold}
        elif args.validation == 'deep':
            validator = DeepLearningValidation(model_name=args.model)
            threshold = {'cosine_similarity': args.similarity_threshold}
        
        # Calculate reference features
        print(f"Calculating reference features using {args.validation}...")
        references = validator.calculate_references(images)
        
        # Validate and filter images
        valid_images = []
        for i, img in enumerate(images):
            # First few images are considered valid as they were used for reference
            if i < len(next(iter(references.values()))):
                valid_images.append(img)
                continue
                
            if validator.validate(img, references, threshold):
                valid_images.append(img)
                print(f"Image {i+1}/{len(images)}: Valid")
            else:
                print(f"Image {i+1}/{len(images)}: Invalid (filtered out)")
        
        print(f"{len(valid_images)} out of {len(images)} images passed validation.")
        
        if len(valid_images) < 2:
            print("Error: Not enough valid images for stitching after filtering.")
            return
        
        # Perform stitching
        print("Stitching images...")
        result = stitch_images(valid_images)
        cv2.imwrite(args.output, result)
        print(f"Stitched image saved to '{args.output}'.")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
