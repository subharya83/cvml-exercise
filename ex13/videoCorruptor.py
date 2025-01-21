import cv2
import numpy as np
import random
import argparse

class VideoCorruptor:
    def __init__(self, video_path, corruption_rate=0.00005, color=(255, 255, 255)):
        self.video_path = video_path
        self.corruption_rate = corruption_rate
        self.corruption_color = color

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open video file.")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.corruption_map = {}

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

    def process_video(self, output_path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        if not out.isOpened():
            raise RuntimeError("Error: Could not open video writer.")

        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            corrupted_frame = self.corrupt_frame(frame, frame_idx)
            out.write(corrupted_frame)
            frame_idx += 1

        out.release()

    def generate_corruption_locations(self):
        total_pixels = self.width * self.height
        num_corrupted_pixels = int(total_pixels * self.corruption_rate)

        locations = []
        while len(locations) < num_corrupted_pixels:
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)

            neighborhood = self.get_valid_neighborhood(x, y)
            num_contiguous = min(5, len(neighborhood))
            random.shuffle(neighborhood)

            for i in range(num_contiguous):
                if len(locations) < num_corrupted_pixels:
                    locations.append(neighborhood[i])

        return locations

    def get_valid_neighborhood(self, x, y):
        neighborhood = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                new_x = x + dx
                new_y = y + dy
                if 0 <= new_x < self.width and 0 <= new_y < self.height:
                    neighborhood.append((new_x, new_y))
        return neighborhood

    def corrupt_frame(self, frame, frame_idx):
        corrupted_frame = frame.copy()

        # Clean up expired corruptions
        self.corruption_map = {
            key: value for key, value in self.corruption_map.items() if value["end_frame"] >= frame_idx
        }

        # Generate new corruptions if needed
        if frame_idx % 5 == 0:
            new_locations = self.generate_corruption_locations()
            for loc in new_locations:
                key = f"{loc[0]},{loc[1]}"
                if key not in self.corruption_map:
                    duration = random.randint(5, 10)
                    self.corruption_map[key] = {
                        "rgb": self.corruption_color,
                        "end_frame": frame_idx + duration
                    }

        # Apply corruptions
        for key, corruption_info in self.corruption_map.items():
            if corruption_info["end_frame"] >= frame_idx:
                x, y = map(int, key.split(","))
                corrupted_frame[y, x] = corruption_info["rgb"]

        return corrupted_frame


def main():
    parser = argparse.ArgumentParser(description="Video Corruption Tool")
    parser.add_argument("-i", "--input", required=True, help="Input video file")
    parser.add_argument("-o", "--output", required=True, help="Output video file")
    parser.add_argument("-c", "--color", default="111", help="Color code for corruption (e.g., 100 for red, 010 for green, 001 for blue)")
    args = parser.parse_args()

    # Parse color code
    color_code = args.color
    if len(color_code) != 3 or any(c not in "01" for c in color_code):
        print("Invalid color code. Using default color (white).")
        color = (255, 255, 255)
    else:
        color = (
            255 if color_code[0] == "1" else 0,
            255 if color_code[1] == "1" else 0,
            255 if color_code[2] == "1" else 0,
        )

    try:
        corruptor = VideoCorruptor(args.input, corruption_rate=0.00005, color=color)
        corruptor.process_video(args.output)
        print("Video corruption completed successfully!")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()