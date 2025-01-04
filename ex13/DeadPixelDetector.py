import cv2
import numpy as np
from scipy.spatial.distance import mahalanobis
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import json
from datetime import datetime

@dataclass
class PixelDefect:
    """Data class to store information about detected pixel defects."""
    x: int
    y: int
    confidence: float
    start_frame: int
    end_frame: Optional[int] = None
    total_appearances: int = 1

    def to_dict(self) -> Dict:
        """Convert defect information to dictionary format."""
        return {
            'x': self.x,
            'y': self.y,
            'confidence': round(float(self.confidence), 3),
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'total_appearances': self.total_appearances,
            'span': self.end_frame - self.start_frame if self.end_frame else None
        }

class DeadPixelDetector:
    def __init__(self, 
                 distance_threshold: float = 3.0,
                 min_persistence: int = 5,
                 noise_threshold: float = 0.1):
        """
        Initialize the dead pixel detector.
        
        Args:
            distance_threshold: Mahalanobis distance threshold for defect detection
            min_persistence: Minimum number of frames a defect must persist
            noise_threshold: Threshold for filtering out noise
        """
        self.distance_threshold = distance_threshold
        self.min_persistence = min_persistence
        self.noise_threshold = noise_threshold
        self.active_defects: Dict[Tuple[int, int], PixelDefect] = {}
        self.completed_defects: List[PixelDefect] = []
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def compute_difference_matrices(self, frame: np.ndarray) -> List[np.ndarray]:
        """Generate 8 difference matrices for the given frame."""
        directions = [
            (1, 1), (1, 0), (1, -1), (0, -1),
            (-1, -1), (-1, 0), (-1, 1), (0, 1)
        ]
        diff_matrices = []
        
        for dx, dy in directions:
            shifted_frame = np.roll(frame, shift=(dx, dy), axis=(0, 1))
            if dx > 0:
                shifted_frame[0, :] = frame[0, :]
            elif dx < 0:
                shifted_frame[-1, :] = frame[-1, :]
            if dy > 0:
                shifted_frame[:, 0] = frame[:, 0]
            elif dy < 0:
                shifted_frame[:, -1] = frame[:, -1]
                
            diff = cv2.absdiff(frame, shifted_frame)
            diff_matrices.append(diff)
            
        return diff_matrices

    def compute_gradient_matrices(self, diff_matrices: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute first and second degree gradient matrices."""
        first_degree = np.stack(diff_matrices[:4], axis=-1)
        second_degree = np.stack(diff_matrices, axis=-1)
        
        for i in range(first_degree.shape[-1]):
            first_degree[..., i] = cv2.GaussianBlur(first_degree[..., i], (3, 3), 0.5)
        for i in range(second_degree.shape[-1]):
            second_degree[..., i] = cv2.GaussianBlur(second_degree[..., i], (3, 3), 0.5)
            
        return first_degree, second_degree

    def detect_defective_pixels(self, frame: np.ndarray) -> List[PixelDefect]:
        """Detect defective pixels in the frame."""
        diff_matrices = self.compute_difference_matrices(frame)
        first_grad, second_grad = self.compute_gradient_matrices(diff_matrices)
        
        candidates = []
        h, w = frame.shape
        
        y_coords, x_coords = np.mgrid[2:h-2, 2:w-2]
        
        for y, x in zip(y_coords.flatten(), x_coords.flatten()):
            first_grad_vector = first_grad[y, x, :].flatten()
            second_grad_vector = second_grad[y, x, :].flatten()
            
            if np.mean(first_grad_vector) < self.noise_threshold:
                continue
                
            try:
                covariance_matrix = np.cov(second_grad_vector, rowvar=False)
                if np.linalg.det(covariance_matrix) < 1e-10:
                    continue
                
                mean_diff = first_grad_vector - np.mean(second_grad_vector)
                distance = mahalanobis(
                    mean_diff,
                    np.mean(second_grad_vector),
                    np.linalg.inv(covariance_matrix)
                )
                
                if distance > self.distance_threshold:
                    candidates.append(PixelDefect(x=x, y=y, confidence=distance, start_frame=0))
                    
            except np.linalg.LinAlgError:
                continue
                
        return candidates

    def update_defects(self, current_defects: List[PixelDefect], frame_number: int):
        """Update tracking of dead pixels across frames."""
        # Check current defects against active defects
        current_coords = set()
        for current in current_defects:
            matched = False
            current_coord = (current.x, current.y)
            current_coords.add(current_coord)
            
            if current_coord in self.active_defects:
                defect = self.active_defects[current_coord]
                defect.total_appearances += 1
                defect.confidence = max(defect.confidence, current.confidence)
                matched = True
                
            if not matched:
                current.start_frame = frame_number
                self.active_defects[current_coord] = current

        # Check for ended defects
        ended_coords = []
        for coord, defect in self.active_defects.items():
            if coord not in current_coords:
                if defect.total_appearances >= self.min_persistence:
                    defect.end_frame = frame_number - 1
                    self.completed_defects.append(defect)
                ended_coords.append(coord)
                
        for coord in ended_coords:
            del self.active_defects[coord]

    def save_results(self, output_path: str):
        """Save detection results to JSON file."""
        # Finalize any remaining active defects
        for defect in self.active_defects.values():
            if defect.total_appearances >= self.min_persistence:
                defect.end_frame = defect.start_frame + defect.total_appearances - 1
                self.completed_defects.append(defect)

        # Prepare results
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'settings': {
                    'distance_threshold': self.distance_threshold,
                    'min_persistence': self.min_persistence,
                    'noise_threshold': self.noise_threshold
                }
            },
            'defects': [defect.to_dict() for defect in self.completed_defects]
        }

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"Results saved to {output_path}")
        self.logger.info(f"Total dead pixels detected: {len(self.completed_defects)}")

    def process_video(self,
                     input_path: str,
                     json_output_path: str,
                     video_output_path: Optional[str] = None):
        """
        Process video and detect dead pixels.
        
        Args:
            input_path: Path to input video
            json_output_path: Path to save JSON results
            video_output_path: Optional path to save visualization video
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
            
        cap = cv2.VideoCapture(str(input_path))
        out = None
        
        if video_output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
            
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Processing video: {input_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_defects = self.detect_defective_pixels(gray_frame)
            self.update_defects(current_defects, frame_count)
            
            if video_output_path:
                # Visualize active and completed defects
                for defect in self.active_defects.values():
                    confidence_color = np.clip(defect.confidence / 10.0, 0, 1)
                    color = (
                        0,
                        int(255 * (1 - confidence_color)),
                        int(255 * confidence_color)
                    )
                    cv2.circle(frame, (defect.x, defect.y), 2, color, -1)
                out.write(frame)
            
            frame_count += 1
            if frame_count % 100 == 0:
                self.logger.info(f"Processed frame {frame_count}/{total_frames}")
                
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Save results to JSON
        self.save_results(json_output_path)

def main():
    """Example usage of the DeadPixelDetector."""
    detector = DeadPixelDetector(
        distance_threshold=3.0,
        min_persistence=5,
        noise_threshold=0.1
    )
    
    try:
        detector.process_video(
            input_path="input_video.mp4",
            json_output_path="dead_pixels_report.json",
            video_output_path="output_video.mp4"  # Optional
        )
    except Exception as e:
        logging.error(f"Error processing video: {e}")

if __name__ == "__main__":
    main()