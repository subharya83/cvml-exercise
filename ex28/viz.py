import json
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def parse_poses(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            transform = np.array(data['transform']).reshape(4, 4)
            position = transform[:3, 3]
            rotation = transform[:3, :3]
            poses.append({
                'position': position,
                'rotation': rotation,
                'position_delta': data['position_delta']
            })
    return poses

def plot_poses(ax, poses, color, label):
    positions = np.array([p['position'] for p in poses])
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            color=color, label=label, marker='o', markersize=3)
    
    # Draw coordinate axes for key poses
    for i in range(0, len(poses), max(1, len(poses)//5)):
        pos = poses[i]['position']
        rot = poses[i]['rotation']
        axis_length = 0.1
        for j, (col, axis) in enumerate(zip(['r', 'g', 'b'], [0, 1, 2])):
            ax.quiver(pos[0], pos[1], pos[2], 
                     rot[0, axis], rot[1, axis], rot[2, axis],
                     length=axis_length, color=col, alpha=0.5)

def main(file1, file2):
    poses1 = parse_poses(file1)
    poses2 = parse_poses(file2)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    plot_poses(ax, poses1, 'blue', 'Poses A')
    plot_poses(ax, poses2, 'green', 'Poses B')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Pose Visualization')
    ax.legend()
    
    # Equal aspect ratio
    max_range = np.array([
        ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
    ).max() / 2.0
    mid_x = (ax.get_xlim()[0] + ax.get_xlim()[1]) * 0.5
    mid_y = (ax.get_ylim()[0] + ax.get_ylim()[1]) * 0.5
    mid_z = (ax.get_zlim()[0] + ax.get_zlim()[1]) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 viz.py poses_a.json poses_b.json")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])
