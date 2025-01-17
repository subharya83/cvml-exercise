import torch
import torch.nn as nn
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm import tqdm

class GaussianSplat:
    def __init__(self, sigma=0.1, min_points=10):
        self.sigma = sigma
        self.min_points = min_points
        
    def _compute_covariance(self, points):
        """Compute the covariance matrix for a set of points."""
        if len(points) < self.min_points:
            return None
        
        mean = np.mean(points, axis=0)
        centered = points - mean
        cov = np.dot(centered.T, centered) / (len(points) - 1)
        return mean, cov
    
    def _gaussian_weight(self, dist, sigma):
        """Compute Gaussian weights based on distance."""
        return np.exp(-dist**2 / (2 * sigma**2))
    
    def process_point_cloud(self, points, colors, num_neighbors=50):
        """Process point cloud using Gaussian splatting."""
        kdtree = cKDTree(points)
        refined_points = []
        refined_colors = []
        
        for i in tqdm(range(len(points)), desc="Processing points"):
            # Find nearest neighbors
            distances, indices = kdtree.query(points[i], k=num_neighbors)
            
            # Compute weights
            weights = self._gaussian_weight(distances, self.sigma)
            
            # Get neighborhood points and their colors
            neighborhood = points[indices]
            neighborhood_colors = colors[indices]
            
            # Compute weighted mean and covariance
            result = self._compute_covariance(neighborhood)
            if result is None:
                continue
                
            mean, cov = result
            
            # Compute weighted color
            weighted_color = np.average(neighborhood_colors, weights=weights, axis=0)
            
            # Add refined point and color
            refined_points.append(mean)
            refined_colors.append(weighted_color)
            
        return np.array(refined_points), np.array(refined_colors)

class ColorOptimizer(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.colors = nn.Parameter(torch.rand(num_points, 3))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self):
        return self.sigmoid(self.colors)

def optimize_colors(points, initial_colors, frames, camera_positions, num_iterations=1000):
    """Optimize colors using video frames."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize color optimizer
    color_optimizer = ColorOptimizer(len(points)).to(device)
    optimizer = torch.optim.Adam(color_optimizer.parameters(), lr=0.01)
    
    # Convert frames to tensors
    frame_tensors = [torch.FloatTensor(frame).to(device) / 255.0 for frame in frames]
    camera_positions = torch.FloatTensor(camera_positions).to(device)
    points = torch.FloatTensor(points).to(device)
    
    # Training loop
    pbar = tqdm(range(num_iterations), desc="Optimizing colors")
    for iteration in pbar:
        optimizer.zero_grad()
        
        # Get current color predictions
        predicted_colors = color_optimizer()
        
        # Compute loss across all frames
        total_loss = 0
        for frame_idx, frame in enumerate(frame_tensors):
            # Project points to image plane (simplified projection)
            camera_pos = camera_positions[frame_idx]
            directions = points - camera_pos
            
            # Simple perspective projection
            focal_length = 1000.0
            projected_points = focal_length * directions[..., :2] / directions[..., 2:3]
            
            # Convert to pixel coordinates
            h, w = frame.shape[:2]
            pixel_coords = torch.stack([
                (projected_points[..., 0] + w/2),
                (projected_points[..., 1] + h/2)
            ], dim=-1)
            
            # Sample colors from frame
            valid_mask = (pixel_coords[..., 0] >= 0) & (pixel_coords[..., 0] < w) & \
                        (pixel_coords[..., 1] >= 0) & (pixel_coords[..., 1] < h)
            
            if valid_mask.any():
                pixel_coords = pixel_coords[valid_mask].long()
                sampled_colors = frame[pixel_coords[..., 1], pixel_coords[..., 0]]
                total_loss += torch.nn.functional.mse_loss(
                    predicted_colors[valid_mask],
                    sampled_colors
                )
        
        # Backward and optimize
        total_loss.backward()
        optimizer.step()
        
        pbar.set_postfix({'loss': total_loss.item()})
    
    return color_optimizer().detach().cpu().numpy()

def enhance_reconstruction(video_path, ply_path, output_path):
    # Load input data
    print("Loading input data...")
    point_cloud = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    
    # Load video frames
    frames = []
    camera_positions = []
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Estimate camera position (simplified)
        frame_idx = len(frames) - 1
        theta = 2 * np.pi * frame_idx / cap.get(cv2.CAP_PROP_FRAME_COUNT)
        camera_positions.append([2 * np.cos(theta), 0, 2 * np.sin(theta)])
    
    cap.release()
    
    # Apply Gaussian splatting
    print("Applying Gaussian splatting...")
    splatter = GaussianSplat(sigma=0.05, min_points=10)
    refined_points, refined_colors = splatter.process_point_cloud(points, colors)
    
    # Optimize colors using video frames
    print("Optimizing colors...")
    optimized_colors = optimize_colors(
        refined_points,
        refined_colors,
        frames,
        np.array(camera_positions)
    )
    
    # Save refined point cloud
    print("Saving refined point cloud...")
    refined_cloud = o3d.geometry.PointCloud()
    refined_cloud.points = o3d.utility.Vector3dVector(refined_points)
    refined_cloud.colors = o3d.utility.Vector3dVector(optimized_colors)
    
    # Optional: Estimate normals
    refined_cloud.estimate_normals()
    
    # Save the result
    o3d.io.write_point_cloud(output_path, refined_cloud)
    print(f"Refined point cloud saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', required=True, help='Input video path')
    parser.add_argument('-p', required=True, help='Input PLY file path')
    parser.add_argument('-o', required=True, help='Output PLY file path')
    
    args = parser.parse_args()
    enhance_reconstruction(args.v, args.p, args.o)