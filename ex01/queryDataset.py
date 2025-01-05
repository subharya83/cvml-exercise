import os
from PIL import Image
import argparse
import json
from embeddings import Embeddings
import faiss
import numpy as np
from annoy import AnnoyIndex

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
    query_embedding = emb_gen.generate_embedding(qi).astype('float32')
    
    # Determine the index format based on the file extension
    if index_file.endswith('.faiss'):
        # Load FAISS index
        index = faiss.read_index(index_file)
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = index.search(query_embedding, 1)
        best_frame_index = indices[0][0]
        best_similarity = 1 - distances[0][0]  # Convert L2 distance to similarity
    elif index_file.endswith('.ann'):
        # Load Annoy index
        dimension = len(query_embedding)
        index = AnnoyIndex(dimension, 'euclidean')
        index.load(index_file)
        best_frame_index = index.get_nns_by_vector(query_embedding, 1, include_distances=True)[0][0]
        best_similarity = 1 - index.get_nns_by_vector(query_embedding, 1, include_distances=True)[1][0]  # Convert Euclidean distance to similarity
    else:
        # Load JSON index
        with open(index_file, 'r') as f:
            frame_index = json.load(f)
        best_similarity = -1
        best_frame_index = 0
        for i, frame_info in enumerate(frame_index):
            similarity = emb_gen.compute_similarity(query_embedding, frame_info['embedding'])
            if similarity > best_similarity:
                best_similarity = similarity
                best_frame_index = i
    
    # Load frame index
    if not index_file.endswith('.json'):
        with open(index_file.replace('.faiss', '.json').replace('.ann', '.json'), 'r') as f:
            frame_index = json.load(f)
    else:
        frame_index = json.load(open(index_file, 'r'))
    
    # Get the most similar frame
    best_frame = frame_index[best_frame_index]
    
    return best_frame, best_similarity

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Search for most similar video frame')
    parser.add_argument('-q', required=True, help='Path to query image')
    parser.add_argument('-i', required=True, help='Path to index [.faiss|.ann|.json]')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Perform search
    best_frame, similarity = search_most_similar_frame(args.q, args.i)
    
    # Print results
    print(f"Most Similar Frame:")
    print(f"Frame Number: {best_frame['frame_number']}")
    print(f"Similarity: {similarity}")

if __name__ == '__main__':
    main()
