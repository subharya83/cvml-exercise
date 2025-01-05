import os
import argparse
import cv2
from PIL import Image
from embeddings import Embeddings
import json
import faiss
import numpy as np
from annoy import AnnoyIndex

def process_input(input_sequence, output_index):
    """
    Process video and generate embeddings for sampled frames.
    
    :param input_sequence: Path to input video file or directory containing images
    :param output_index: Path to output index file
    """
    # Initialize embedding generator
    emb_gen = Embeddings(model_name='resnet50', download_weights=True)
    
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
    embeddings_list = []
    
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
                if frame_count == len(seq):
                    break
                filename = os.path.join(input_sequence, seq[frame_count])
                frame = Image.open(filename).convert('RGB')

            e = emb_gen.generate_embedding(frame)
            # Store frame information
            frame_info = {
                'frame_number': frame_count,
                'embedding': e.tolist()
            }
            frame_index.append(frame_info)
            embeddings_list.append(e)
            frame_count += 1

    finally:
        # Release video capture
        if isVideo:
            print('Video processed')
            seq.release()
        else:
            print('Image sequence processed')
    
    # Convert embeddings list to numpy array
    embeddings_array = np.array(embeddings_list).astype('float32')
    
    # Determine the index format based on the output file extension
    if output_index.endswith('.faiss'):
        # Build FAISS index
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)  # Using L2 distance for NN search
        index.add(embeddings_array)
        faiss.write_index(index, output_index)
        print(f"FAISS index saved to {output_index}")
    elif output_index.endswith('.ann'):
        # Build Annoy index
        dimension = embeddings_array.shape[1]
        index = AnnoyIndex(dimension, 'euclidean')  # Using Euclidean distance for NN search
        for i, embedding in enumerate(embeddings_array):
            index.add_item(i, embedding)
        index.build(10)  # 10 trees
        index.save(output_index)
        print(f"Annoy index saved to {output_index}")
    else:
        # Save as JSON
        with open(output_index, 'w') as f:
            json.dump(frame_index, f)
        print(f"JSON index saved to {output_index}")
    
    print(f"Processed {len(frame_index)} frames.")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Generate embeddings for video frames')
    parser.add_argument('-i', help='Path to input video or directory containing images')
    parser.add_argument('-o', help='Path to output embedding index file [.faiss|.ann|.json]')
   
    # Parse arguments
    args = parser.parse_args()
    
    # Process video
    process_input(args.i, args.o)

if __name__ == '__main__':
    main()
