import os
from PIL import Image
import argparse
import json
from embeddings import Embeddings

def search_most_similar_frame(query_image, index_file):
    """
    Find the most similar frame from the video index.
    
    :param query_image: Path to the query image
    :param index_file: Path to the video frame embedding index
    :return: Information about the most similar frame
    """
    # Initialize embedding generator
    emb_gen = Embeddings(model_name='resnet50', download_weights=True)
    
    # Generate embedding for query image
    qi = Image.open(query_image).convert('RGB')
    query_embedding = emb_gen.generate_embedding(qi)
    
    # Load frame index
    with open(index_file, 'r') as f:
        frame_index = json.load(f)
    
    # Find most similar frame
    best_similarity = -1
    best_frame = None
    
    for frame_info in frame_index:
        # Compute similarity
        similarity = emb_gen.compute_cosine_similarity(query_embedding, frame_info['embedding'])
        
        # Update best match
        if similarity > best_similarity:
            best_similarity = similarity
            best_frame = frame_info
    
    return best_frame, best_similarity

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Search for most similar video frame')
    parser.add_argument('-q', required=True, help='Path to query image')
    parser.add_argument('-i', required=True, help='Path to index')
    
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