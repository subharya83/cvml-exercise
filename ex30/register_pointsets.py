import numpy as np
import open3d as o3d
import csv
import argparse
from sklearn.neighbors import NearestNeighbors

def load_points_from_csv(filename):
    """Load points from CSV file (x,y,z format)"""
    points = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            points.append([float(row[0]), float(row[1]), float(row[2])])
    return np.array(points)

def compute_fpfh_features(pcd, radius=1.0):
    """Compute FPFH features for global registration"""
    radius_normal = radius * 2
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_normal, max_nn=30))
    
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100))

def fast_global_registration(source, target, source_fpfh, target_fpfh, distance_threshold=0.05):
    """Fast Global Registration algorithm"""
    return o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))

def multi_scale_icp(source, target, voxel_sizes=[2.0, 1.0, 0.5], max_iters=[100, 50, 30]):
    """Multi-scale ICP registration"""
    current_transformation = np.identity(4)
    for i, (voxel_size, max_iter) in enumerate(zip(voxel_sizes, max_iters)):
        print(f"Scale {i+1}: voxel_size={voxel_size}, max_iter={max_iter}")
        
        # Downsample
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)
        
        # Estimate normals
        radius_normal = voxel_size * 2
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        # Run ICP
        result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, voxel_size * 1.4,
            current_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
        
        current_transformation = result.transformation
    
    return current_transformation

def register_point_sets(source_points, target_points, method='fgr+icp', verbose=True):
    """Improved registration pipeline with multiple algorithm options"""
    # Convert to Open3D point clouds
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)
    
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_points)
    
    methods = {
        'icp': lambda: o3d.pipelines.registration.registration_icp(
            source, target, 1.0, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)),
        
        'fgr': lambda: fast_global_registration(
            source, target, 
            compute_fpfh_features(source), 
            compute_fpfh_features(target)),
        
        'fgr+icp': lambda: (
            fgr_result := fast_global_registration(
                source, target, 
                compute_fpfh_features(source), 
                compute_fpfh_features(target)),
            o3d.pipelines.registration.registration_icp(
                source, target, 0.05, fgr_result.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))),
        
        'multi_scale': lambda: (
            transform := multi_scale_icp(source, target),
            o3d.pipelines.registration.evaluate_registration(
                source, target, 0.05, transform))
    }
    
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Choose from {list(methods.keys())}")
    
    result = methods[method]()
    
    if method == 'multi_scale':
        transformation = result[0]
        result = result[1]
    else:
        transformation = result.transformation
    
    # Apply final transformation
    registered_source = source.transform(transformation)
    
    # Compute additional metrics
    distances = np.linalg.norm(np.asarray(registered_source.points) - np.asarray(target.points), axis=1)
    mean_distance = np.mean(distances)
    median_distance = np.median(distances)
    
    if verbose:
        print(f"\nRegistration Method: {method.upper()}")
        print(f"Fitness: {result.fitness:.4f} (1.0 is perfect)")
        print(f"Inlier RMSE: {result.inlier_rmse:.6f}")
        print(f"Mean Distance: {mean_distance:.6f}")
        print(f"Median Distance: {median_distance:.6f}")
    
    return {
        'transformation': transformation,
        'registered_points': np.asarray(registered_source.points),
        'fitness': result.fitness,
        'inlier_rmse': result.inlier_rmse,
        'mean_distance': mean_distance,
        'median_distance': median_distance
    }

def main():
    parser = argparse.ArgumentParser(description='Improved point set registration')
    parser.add_argument('source_csv', help='Source point cloud CSV')
    parser.add_argument('target_csv', help='Target point cloud CSV')
    parser.add_argument('--method', default='fgr+icp', 
                       help='Registration method: icp, fgr, fgr+icp, multi_scale')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    
    args = parser.parse_args()
    
    # Load point sets
    source_points = load_points_from_csv(args.source_csv)
    target_points = load_points_from_csv(args.target_csv)
    
    # Perform registration
    result = register_point_sets(source_points, target_points, args.method)
    
    # Visualization
    if args.visualize:
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(source_points)
        source.paint_uniform_color([1, 0, 0])  # Red
        
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(target_points)
        target.paint_uniform_color([0, 1, 0])  # Green
        
        registered = o3d.geometry.PointCloud()
        registered.points = o3d.utility.Vector3dVector(result['registered_points'])
        registered.paint_uniform_color([0, 0, 1])  # Blue
        
        o3d.visualization.draw_geometries(
            [source, target, registered, 
             o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)],
            window_name=f"Registration Result ({args.method.upper()})",
            width=800, height=600)

if __name__ == "__main__":
    main()
