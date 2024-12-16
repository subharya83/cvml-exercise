import argparse
from embeddings import Embeddings

# Example usage
def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Get similarity between two images')
    parser.add_argument('-q', required=True, help='Path to query image')
    parser.add_argument('-t', required=True, help='Path to test image')
    
    # Parse arguments
    args = parser.parse_args()
    # Create an embedding generator with local weight download
    embedding_generator = Embeddings(
        model_name='resnet50',  # or 'resnet18'
        download_weights=True
    )
    
    # Generate embeddings for two images
    qe = embedding_generator.generate_embedding(args.q)
    te = embedding_generator.generate_embedding(args.t)
    
    # Compute similarity
    similarity = embedding_generator.compute_similarity(qe, te)
    print(f"Image Similarity: {similarity}")

if __name__ == '__main__':
    main()
