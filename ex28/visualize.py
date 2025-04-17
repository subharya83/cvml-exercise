import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the JSON output file
def load_transforms(json_file):
    # Read the file and split it into individual JSON objects
    with open(json_file, 'r') as f:
        content = f.read()
    
    # Handle the non-standard JSON format (separate JSON objects)
    json_objects = []
    for line in content.strip().split('\n'):
        json_objects.append(json.loads(line))
    
    return json_objects

# Extract camera positions from transforms
def extract_camera_positions(transforms):
    positions = []
    for transform_data in transforms:
        transform = np.array(transform_data['transform']).reshape(4, 4)
        # Camera position is in the last column of the inverse transform
        # Since we're dealing with rigid transforms, we can just use the transpose of rotation part
        R = transform[:3, :3]
        t = transform[:3, 3]
        camera_position = -np.dot(R.T, t)  # C = -R^T * t
        positions.append(camera_position)
    
    return np.array(positions)

# Visualize the camera path
def visualize_camera_path(positions, ax):
    # Plot the camera path
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', marker='o', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', marker='o', s=100, label='End')
    
    # Add some camera frustums at regular intervals (simplified as points for clarity)
    stride = max(1, len(positions) // 10)  # Show at most 10 frustums
    for i in range(0, len(positions), stride):
        if i != 0 and i != len(positions) - 1:  # Skip start and end (already marked)
            ax.scatter(positions[i, 0], positions[i, 1], positions[i, 2], c='black', marker='.', s=30)
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.set_title('3D Camera Path Visualization', fontsize=12)
    ax.legend(fontsize=9)
    
    # Set equal aspect ratio
    max_range = np.max([
        np.ptp(positions[:, 0]),
        np.ptp(positions[:, 1]),
        np.ptp(positions[:, 2])
    ])
    mid_x = np.mean(positions[:, 0])
    mid_y = np.mean(positions[:, 1])
    mid_z = np.mean(positions[:, 2])
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

# Visualize the position deltas
def visualize_position_deltas(transforms, ax):
    deltas = [t['position_delta'] for t in transforms]
    
    ax.plot(deltas, linewidth=2)
    ax.set_title('Camera Position Delta Over Frames', fontsize=12)
    ax.set_xlabel('Frame Number', fontsize=10)
    ax.set_ylabel('Position Delta', fontsize=10)
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=9)

def main():
    parser = argparse.ArgumentParser(description='Visualize camera poses from json files')
    parser.add_argument('-i', required=True, help='Input pose file')
    parser.add_argument('-o', required=True, help='Path to the output image file')
    args = parser.parse_args()

    transforms = load_transforms(args.i)
    positions = extract_camera_positions(transforms)
    
    # Create a single figure with two subplots
    plt.figure(figsize=(14, 6))
    
    # First subplot for 3D camera path
    ax1 = plt.subplot(121, projection='3d')
    visualize_camera_path(positions, ax1)
    
    # Second subplot for position deltas
    ax2 = plt.subplot(122)
    visualize_position_deltas(transforms, ax2)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(args.o, dpi=150)
    plt.close()
    
if __name__ == "__main__":
    main()
    