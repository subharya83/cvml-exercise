import open3d as o3d
import numpy as np
import argparse
from pathlib import Path

def load_point_cloud(file_path):
    """
    Load a PLY file and return point cloud object
    """
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if not pcd.has_points():
            raise ValueError("Loaded point cloud has no points")
        return pcd
    except Exception as e:
        raise Exception(f"Error loading point cloud: {str(e)}")

def process_point_cloud(pcd, voxel_size=0.02):
    """
    Process and refine the point cloud:
    1. Statistical outlier removal
    2. Estimate normals
    3. Uniform downsampling
    """
    # Statistical outlier removal
    points = np.asarray(pcd.points)
    if len(points) == 0:
        raise ValueError("Point cloud is empty")

    # Voxel downsampling first to reduce computation time
    print("Performing voxel downsampling...")
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Estimate normals
    print("Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # Simple statistical filtering based on distance to neighbors
    print("Performing statistical filtering...")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)
    filtered_points = []
    
    for i in range(len(points)):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], 20)
        if k < 3:  # Skip points with too few neighbors
            continue
            
        # Calculate mean distance to neighbors
        distances = np.linalg.norm(points[idx[1:]] - points[i], axis=1)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # Keep points within 2 standard deviations
        if mean_dist <= np.mean(distances) + 2 * std_dist:
            filtered_points.append(points[i])
    
    # Create new point cloud with filtered points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(np.array(filtered_points))
    
    # Re-estimate normals for the filtered cloud
    filtered_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    return filtered_pcd

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process and refine a PLY point cloud file")
    parser.add_argument("-i", help="Path to input PLY file")
    parser.add_argument("-o", help="Path to save processed PLY file")
    parser.add_argument("-s", type=float, default=0.02,
                        help="Voxel size for downsampling (default: 0.2)")
    
    args = parser.parse_args()
    
    try:
        # Check if input file exists
        if not Path(args.i).exists():
            raise FileNotFoundError(f"Input file {args.i} not found")
            
        # Load point cloud
        print(f"Loading point cloud from {args.i}")
        pcd = load_point_cloud(args.i)
        
        # Process point cloud
        print("Processing point cloud...")
        print(f"Original point cloud has {len(pcd.points)} points")
        processed_pcd = process_point_cloud(pcd, args.s)
        print(f"Processed point cloud has {len(processed_pcd.points)} points")
        
        # Save processed point cloud
        print(f"Saving processed point cloud to {args.o}")
        o3d.io.write_point_cloud(args.o, processed_pcd)
        print("Processing completed successfully")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())