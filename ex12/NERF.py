import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
from tqdm import tqdm

class NeRFNetwork(nn.Module):
    def __init__(self, pos_dim=3, dir_dim=3, hidden_dim=256):
        super().__init__()
        
        # Position encoding network
        self.pos_network = nn.Sequential(
            nn.Linear(pos_dim + 60, hidden_dim),  # 60 is from positional encoding
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Direction encoding network
        self.dir_network = nn.Sequential(
            nn.Linear(dir_dim + hidden_dim + 24, hidden_dim//2),  # 24 is from directional encoding
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 3),  # RGB output
            nn.Sigmoid(),
        )
        
        self.density_layer = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def positional_encoding(self, x, num_frequencies=10):
        encodings = [x]
        for i in range(num_frequencies):
            for fn in [torch.sin, torch.cos]:
                encodings.append(fn(2.0 ** i * x))
        return torch.cat(encodings, dim=-1)

    def forward(self, positions, directions):
        # Encode positions and directions
        encoded_pos = self.positional_encoding(positions, num_frequencies=10)
        encoded_dir = self.positional_encoding(directions, num_frequencies=4)
        
        # Get features from position
        pos_features = self.pos_network(encoded_pos)
        
        # Get density
        density = self.relu(self.density_layer(pos_features))
        
        # Combine features with direction for color prediction
        dir_input = torch.cat([pos_features, encoded_dir], dim=-1)
        color = self.dir_network(dir_input)
        
        return density, color

class NeRFDataset(Dataset):
    def __init__(self, video_path, ply_path):
        self.ply_cloud = o3d.io.read_point_cloud(ply_path)
        self.points = np.asarray(self.ply_cloud.points)
        self.colors = np.asarray(self.ply_cloud.colors)
        
        # Load video frames and camera parameters
        self.frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        # Calculate camera parameters (simplified)
        self.focal_length = max(self.frames[0].shape[:2])
        self.camera_positions = self._estimate_camera_positions()

    def _estimate_camera_positions(self):
        # Simplified camera position estimation
        num_frames = len(self.frames)
        radius = 2.0
        positions = []
        for i in range(num_frames):
            theta = 2 * np.pi * i / num_frames
            pos = np.array([radius * np.cos(theta), 0.0, radius * np.sin(theta)])
            positions.append(pos)
        return np.array(positions)

    def __len__(self):
        return len(self.points) * len(self.frames)

    def __getitem__(self, idx):
        point_idx = idx % len(self.points)
        frame_idx = idx // len(self.points)
        
        position = self.points[point_idx]
        color = self.colors[point_idx]
        camera_pos = self.camera_positions[frame_idx]
        
        # Calculate viewing direction
        direction = position - camera_pos
        direction = direction / np.linalg.norm(direction)
        
        return {
            'position': torch.FloatTensor(position),
            'direction': torch.FloatTensor(direction),
            'color': torch.FloatTensor(color)
        }

def train_nerf(video_path, ply_path, output_path, num_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize dataset and model
    dataset = NeRFDataset(video_path, ply_path)
    dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)
    model = NeRFNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in pbar:
            positions = batch['position'].to(device)
            directions = batch['direction'].to(device)
            target_colors = batch['color'].to(device)
            
            optimizer.zero_grad()
            
            density, predicted_colors = model(positions, directions)
            loss = torch.nn.functional.mse_loss(predicted_colors, target_colors)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
    
    # Save refined point cloud
    refined_points = []
    refined_colors = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Generating refined point cloud'):
            positions = batch['position'].to(device)
            directions = batch['direction'].to(device)
            
            density, colors = model(positions, directions)
            
            mask = density.squeeze() > 0.5
            refined_points.extend(positions[mask].cpu().numpy())
            refined_colors.extend(colors[mask].cpu().numpy())
    
    refined_cloud = o3d.geometry.PointCloud()
    refined_cloud.points = o3d.utility.Vector3dVector(np.array(refined_points))
    refined_cloud.colors = o3d.utility.Vector3dVector(np.array(refined_colors))
    o3d.io.write_point_cloud(output_path, refined_cloud)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', required=True, help='Input video path')
    parser.add_argument('-p', required=True, help='Input PLY file path')
    parser.add_argument('-o', required=True, help='Output PLY file path')
    parser.add_argument('-e', type=int, default=50, help='Number of training epochs')
    
    args = parser.parse_args()
    train_nerf(args.v, args.p, args.o, args.e)