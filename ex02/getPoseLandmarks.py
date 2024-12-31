import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm

class PoseTrajectoryTracker:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Dictionary to store trajectories
        self.trajectories = {
            'frame_number': [],
            'timestamp': []
        }
        
        # Initialize only body joint names (excluding facial landmarks)
        self.joint_names = [
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle'
        ]
        
        # Dictionary mapping joint names to MediaPipe landmark indices
        self.joint_indices = {
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28
        }
        
        # Initialize columns for x, y, z coordinates and visibility
        for joint in self.joint_names:
            self.trajectories[f'{joint}_x'] = []
            self.trajectories[f'{joint}_y'] = []
            self.trajectories[f'{joint}_z'] = []
            self.trajectories[f'{joint}_visibility'] = []

    def process_video(self, video_path):
        """Process video and extract pose trajectories"""
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Error: Could not open video file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process each frame
        with tqdm(total=frame_count, desc="Processing frames") as pbar:
            frame_number = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get timestamp
                timestamp = frame_number / fps
                
                # Process frame
                self.process_frame(frame, frame_number, timestamp)
                
                frame_number += 1
                pbar.update(1)
        
        cap.release()

    def process_frame(self, frame, frame_number, timestamp):
        """Process a single frame and extract pose landmarks"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(image_rgb)
        
        # Store frame number and timestamp
        self.trajectories['frame_number'].append(frame_number)
        self.trajectories['timestamp'].append(timestamp)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Store coordinates for each joint
            for joint in self.joint_names:
                idx = self.joint_indices[joint]
                landmark = landmarks[idx]
                self.trajectories[f'{joint}_x'].append(landmark.x)
                self.trajectories[f'{joint}_y'].append(landmark.y)
                self.trajectories[f'{joint}_z'].append(landmark.z)
                self.trajectories[f'{joint}_visibility'].append(landmark.visibility)
        else:
            # Fill with NaN if no pose is detected
            for joint in self.joint_names:
                self.trajectories[f'{joint}_x'].append(np.nan)
                self.trajectories[f'{joint}_y'].append(np.nan)
                self.trajectories[f'{joint}_z'].append(np.nan)
                self.trajectories[f'{joint}_visibility'].append(0.0)

    def save_trajectories(self, output_path):
        """Save trajectories to CSV file"""
        # Convert to DataFrame
        df = pd.DataFrame(self.trajectories)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Trajectories saved to: {output_path}")
        
        # Print summary statistics
        print("\nTrajectory Summary:")
        print(f"Total frames processed: {len(df)}")
        print(f"Number of joints tracked: {len(self.joint_names)}")
        
        # Calculate average visibility for each joint
        print("\nAverage visibility per joint:")
        for joint in self.joint_names:
            avg_visibility = df[f'{joint}_visibility'].mean()
            print(f"{joint:>15}: {avg_visibility:.2f}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Track human body joint trajectories in video')
    parser.add_argument('-i', type=str, help='Path to input video file')
    parser.add_argument('-o', type=str, default='joint_trajectories.csv',
                        help='Path to output CSV file (default: joint_trajectories.csv)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input video path
    if not Path(args.i).exists():
        raise FileNotFoundError(f"Input video file not found: {args.i}")
    
    # Create tracker and process video
    tracker = PoseTrajectoryTracker()
    print(f"Processing video: {args.i}")
    tracker.process_video(args.i)
    tracker.save_trajectories(args.o)

if __name__ == "__main__":
    main()