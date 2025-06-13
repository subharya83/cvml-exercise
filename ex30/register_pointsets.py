import numpy as np
import open3d as o3d
import csv
import argparse

def load_points_from_csv(filename):
    """Load points from CSV file (x,y,z format)"""
    points = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            points.append([float(row[0]), float(row[1]), float(row[2])])
    return np.array(points)

def register_point_sets(source_points, target_points, threshold=1.0, max_iterations=100):
    """
    Register source points to target points using ICP algorithm
    Returns:
    - transformation matrix
    - registered source points
    - fitness score (0-1 where 1 is best)
    - inlier RMSE
    """
    # Convert numpy arrays to Open3D point clouds
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)
    
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_points)
    
    # Run ICP registration
    reg_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations))
    
    # Apply transformation to source points
    transformed_source = source.transform(reg_result.transformation)
    
    return reg_result.transformation, np.asarray(transformed_source.points), reg_result.fitness, reg_result.inlier_rmse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Register two point sets using ICP algorithm')
    parser.add_argument('source_csv', help='CSV file containing source point cloud')
    parser.add_argument('target_csv', help='CSV file containing target point cloud')
    parser.add_argument('--max_iterations', type=int, default=100, help='Maximum iterations for ICP')
    parser.add_argument('--threshold', type=float, default=1.0, help='Distance threshold for correspondence matching')
    parser.add_argument('--visualize', action='store_true', help='Visualize the registration result')
    
    args = parser.parse_args()
    
    # Load point sets
    source_points = load_points_from_csv(args.source_csv)
    target_points = load_points_from_csv(args.target_csv)
    
    print(f"Source points: {len(source_points)} points")
    print(f"Target points: {len(target_points)} points")
    
    # Perform registration
    transformation, registered_points, fitness, rmse = register_point_sets(
        source_points, target_points, args.threshold, args.max_iterations)
    
    print("\nRegistration Results:")
    print(f"Transformation Matrix:\n{transformation}")
    print(f"Fitness Score: {fitness:.4f} (1.0 is perfect)")
    print(f"Inlier RMSE: {rmse:.6f}")
    
    # Visualization if requested
    if args.visualize:
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_points)
        source_pcd.paint_uniform_color([1, 0, 0])  # Red
        
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_points)
        target_pcd.paint_uniform_color([0, 1, 0])  # Green
        
        registered_pcd = o3d.geometry.PointCloud()
        registered_pcd.points = o3d.utility.Vector3dVector(registered_points)
        registered_pcd.paint_uniform_color([0, 0, 1])  # Blue
        
        # Coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        
        o3d.visualization.draw_geometries([source_pcd, target_pcd, registered_pcd, coord_frame],
                                         window_name="Registration Result",
                                         width=800, height=600)

if __name__ == "__main__":
    main()
