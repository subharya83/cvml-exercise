import cv2
import numpy as np
from typing import Tuple, List
import random
import argparse

class VideoCorruptor:
    def __init__(self, video_path: str, corruption_rate: float = 0.00005):
        """
        Initialize the video corruptor with given parameters.
        
        Args:
            video_path: Path to the input video file
            corruption_rate: Percentage of pixels to corrupt (as decimal)
        """
        self.video_path = video_path
        self.corruption_rate = corruption_rate
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Store corruption locations and their temporal duration
        self.corruption_map = {}
        
    def generate_corruption_locations(self) -> List[Tuple[int, int]]:
        """Generate random locations for pixel corruption."""
        total_pixels = self.width * self.height
        num_corrupted_pixels = int(total_pixels * self.corruption_rate)
        
        locations = []
        while len(locations) < num_corrupted_pixels:
            x = random.randint(1, self.width - 2)  # Avoid edges
            y = random.randint(1, self.height - 2)
            
            # Generate 1-5 contiguous pixels in 3x3 neighborhood
            num_contiguous = random.randint(1, 5)
            neighborhood = self._get_valid_neighborhood(x, y)
            selected_pixels = random.sample(neighborhood, min(num_contiguous, len(neighborhood)))
            
            locations.extend(selected_pixels)
        
        return locations[:num_corrupted_pixels]  # Ensure we don't exceed desired count
    
    def _get_valid_neighborhood(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid pixels in 3x3 neighborhood."""
        neighborhood = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < self.width and 0 <= new_y < self.height):
                    neighborhood.append((new_x, new_y))
        return neighborhood
    
    def corrupt_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Corrupt a single frame based on existing corruption map or generate new corruptions.
        
        Args:
            frame: Input frame
            frame_idx: Current frame index
        
        Returns:
            Corrupted frame
        """
        corrupted_frame = frame.copy()
        
        # Clean up expired corruptions
        self.corruption_map = {k: v for k, v in self.corruption_map.items() 
                             if v['end_frame'] >= frame_idx}
        
        # Generate new corruptions if needed
        if frame_idx % 5 == 0:
            new_locations = self.generate_corruption_locations()
            for loc in new_locations:
                if loc not in self.corruption_map:
                    # Generate random but similar RGB values
                    base_intensity = random.randint(0, 255)
                    variation = random.randint(-20, 20)
                    rgb = [
                        max(0, min(255, base_intensity + variation)),
                        max(0, min(255, base_intensity + variation)),
                        max(0, min(255, base_intensity + variation))
                    ]
                    
                    self.corruption_map[loc] = {
                        'rgb': rgb,
                        'end_frame': frame_idx + random.randint(5, 10)
                    }
        
        # Apply corruptions
        for (x, y), corruption_info in self.corruption_map.items():
            if corruption_info['end_frame'] >= frame_idx:
                corrupted_frame[y, x] = corruption_info['rgb']
        
        return corrupted_frame
    
    def process_video(self, output_path: str):
        """
        Process the entire video and save the corrupted version.
        
        Args:
            output_path: Path to save the corrupted video
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                            (self.width, self.height))
        
        frame_idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            corrupted_frame = self.corrupt_frame(frame, frame_idx)
            out.write(corrupted_frame)
            frame_idx += 1
            
        self.cap.release()
        out.release()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video corruptor to simulate dead pixels')
    parser.add_argument('-i', required=True, help='Path to input video')
    parser.add_argument('-o', required=True, help='Path to corrupted output video')
    
    # Parse arguments
    args = parser.parse_args()
    corruptor = VideoCorruptor(args.i)
    corruptor.process_video(args.o)
