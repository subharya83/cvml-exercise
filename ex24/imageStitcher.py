import os
import argparse
import cv2
import numpy as np

def compute_edge_density(image):
    """Compute edge density using Canny edge detector."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges > 0) / (image.shape[0] * image.shape[1])

def compute_color_histogram(image):
    """Compute color histogram for an image."""
    hist = []
    for i in range(3):
        hist.extend(cv2.calcHist([image], [i], None, [8], [0, 256]).flatten())
    return hist

def validate_image(image, reference_histograms, reference_edge_densities, hist_threshold=0.5, edge_threshold=0.5):
    """
    Validate if an image belongs to the set based on color histogram and edge density.
    
    Args:
        image: The image to validate
        reference_histograms: List of histograms from the reference set
        reference_edge_densities: List of edge densities from the reference set
        hist_threshold: Threshold for histogram comparison (0-1)
        edge_threshold: Threshold for edge density difference (0-1)
        
    Returns:
        True if image is valid, False otherwise
    """
    # Compute histogram for current image
    hist = compute_color_histogram(image)
    
    # Compute edge density for current image
    edge_density = compute_edge_density(image)
    
    # Check if image is similar to at least one reference image
    for ref_hist, ref_edge_density in zip(reference_histograms, reference_edge_densities):
        # Compare histograms using correlation
        hist_similarity = cv2.compareHist(np.array(hist, dtype=np.float32), 
                                         np.array(ref_hist, dtype=np.float32), 
                                         cv2.HISTCMP_CORREL)
        
        # Compare edge densities
        edge_diff = abs(edge_density - ref_edge_density) / max(edge_density, ref_edge_density)
        edge_similarity = 1 - edge_diff
        
        # If image is similar in both aspects, consider it valid
        if hist_similarity > hist_threshold and edge_similarity > edge_threshold:
            return True
            
    return False

def calculate_reference_features(images, sample_size=5):
    """
    Calculate reference features from a subset of images.
    
    Args:
        images: List of loaded images
        sample_size: Number of images to use as reference
        
    Returns:
        Tuple of (reference_histograms, reference_edge_densities)
    """
    if len(images) <= sample_size:
        sample_indices = range(len(images))
    else:
        # Choose random images as reference
        sample_indices = np.random.choice(len(images), sample_size, replace=False)
    
    reference_histograms = []
    reference_edge_densities = []
    
    for idx in sample_indices:
        reference_histograms.append(compute_color_histogram(images[idx]))
        reference_edge_densities.append(compute_edge_density(images[idx]))
    
    return reference_histograms, reference_edge_densities

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

def main():
    parser = argparse.ArgumentParser(description="Stitch multiple images into a panorama with validation")
    parser.add_argument("-i", "--input", required=True, help="Input directory containing images")
    parser.add_argument("-o", "--output", required=True, help="Output file for the stitched panorama")
    parser.add_argument("--hist-threshold", type=float, default=0.6, help="Threshold for histogram similarity (0-1)")
    parser.add_argument("--edge-threshold", type=float, default=0.5, help="Threshold for edge density similarity (0-1)")
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist.")
        return
    
    # Load images
    image_files = [f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        print(f"Error: No images found in '{args.input}'.")
        return
    
    print(f"Found {len(image_files)} images.")
    
    # Load all images
    images = []
    for image_file in image_files:
        image_path = os.path.join(args.input, image_file)
        img = cv2.imread(image_path)
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: Could not load image '{image_file}'.")
    
    if len(images) < 2:
        print("Error: Need at least 2 valid images for stitching.")
        return
    
    # Calculate reference features
    print("Calculating reference features...")
    reference_histograms, reference_edge_densities = calculate_reference_features(images)
    
    # Validate and filter images
    valid_images = []
    for i, img in enumerate(images):
        # First few images are considered valid as they were used for reference
        if i < len(reference_histograms):
            valid_images.append(img)
            continue
            
        if validate_image(img, reference_histograms, reference_edge_densities, 
                         args.hist_threshold, args.edge_threshold):
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
    try:
        result = stitch_images(valid_images)
        cv2.imwrite(args.output, result)
        print(f"Stitched image saved to '{args.output}'.")
    except Exception as e:
        print(f"Stitching failed: {e}")

if __name__ == "__main__":
    main()
