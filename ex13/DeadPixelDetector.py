import cv2
import numpy as np
from scipy.spatial.distance import mahalanobis
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

@dataclass
class PixelDefect:
    """Data class to store information about detected pixel defects."""
    x: int
    y: int
    confidence: float
    persistence: int = 0  # Number of frames the defect has been detected
    
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
        self.persistent_defects = {}  # Track defects across frames
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def compute_difference_matrices(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Generate 8 difference matrices for the given frame.
        
        Args:
            frame: Input grayscale frame
            
        Returns:
            List of difference matrices for 8 directions
        """
        directions = [
            (1, 1), (1, 0), (1, -1), (0, -1),
            (-1, -1), (-1, 0), (-1, 1), (0, 1)
        ]
        diff_matrices = []
        
        for dx, dy in directions:
            shifted_frame = np.roll(frame, shift=(dx, dy), axis=(0, 1))
            # Apply border handling
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
    
    def compute_gradient_matrices(self, 
                                diff_matrices: List[np.ndarray]
                                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute first and second degree gradient matrices.
        
        Args:
            diff_matrices: List of difference matrices
            
        Returns:
            Tuple of first and second degree gradient matrices
        """
        first_degree = np.stack(diff_matrices[:4], axis=-1)
        second_degree = np.stack(diff_matrices, axis=-1)
        
        # Apply Gaussian smoothing to reduce noise
        for i in range(first_degree.shape[-1]):
            first_degree[..., i] = cv2.GaussianBlur(
                first_degree[..., i], (3, 3), 0.5
            )
        for i in range(second_degree.shape[-1]):
            second_degree[..., i] = cv2.GaussianBlur(
                second_degree[..., i], (3, 3), 0.5
            )
            
        return first_degree, second_degree
    
    def detect_defective_pixels(self, 
                              frame: np.ndarray
                              ) -> List[PixelDefect]:
        """
        Detect defective pixels in the frame using statistical gradient analysis.
        
        Args:
            frame: Input grayscale frame
            
        Returns:
            List of detected pixel defects
        """
        diff_matrices = self.compute_difference_matrices(frame)
        first_grad, second_grad = self.compute_gradient_matrices(diff_matrices)
        
        candidates = []
        h, w = frame.shape
        
        # Create meshgrid for vectorized operations
        y_coords, x_coords = np.mgrid[2:h-2, 2:w-2]
        
        for y, x in zip(y_coords.flatten(), x_coords.flatten()):
            first_grad_vector = first_grad[y, x, :].flatten()
            second_grad_vector = second_grad[y, x, :].flatten()
            
            # Skip if gradient vectors are too small (likely uniform region)
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
                    candidates.append(PixelDefect(x, y, distance))
                    
            except np.linalg.LinAlgError:
                continue
                
        return candidates
    
    def update_persistent_defects(self, 
                                current_defects: List[PixelDefect]
                                ) -> List[PixelDefect]:
        """
        Update and track persistent defects across frames.
        
        Args:
            current_defects: List of defects detected in current frame
            
        Returns:
            List of confirmed persistent defects
        """
        # Update existing defects
        for coord, defect in list(self.persistent_defects.items()):
            found = False
            for current in current_defects:
                if (abs(current.x - defect.x) <= 1 and 
                    abs(current.y - defect.y) <= 1):
                    defect.persistence += 1
                    found = True
                    break
            
            if not found:
                defect.persistence -= 1
                if defect.persistence < 0:
                    del self.persistent_defects[coord]
                    
        # Add new defects
        for defect in current_defects:
            coord = (defect.x, defect.y)
            if coord not in self.persistent_defects:
                self.persistent_defects[coord] = defect
                
        # Return only confirmed persistent defects
        return [
            defect for defect in self.persistent_defects.values()
            if defect.persistence >= self.min_persistence
        ]
    
    def process_video(self,
                     input_path: str,
                     output_path: Optional[str] = None,
                     display: bool = False):
        """
        Process the video and detect dead pixels.
        
        Args:
            input_path: Path to input video
            output_path: Optional path to save processed video
            display: Whether to display the processed frames
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
            
        cap = cv2.VideoCapture(str(input_path))
        out = None
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Processing video: {input_path}")
        self.logger.info(f"Total frames: {total_frames}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 100 == 0:
                self.logger.info(f"Processing frame {frame_count}/{total_frames}")
                
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            candidates = self.detect_defective_pixels(gray_frame)
            persistent_defects = self.update_persistent_defects(candidates)
            
            # Visualize defects
            for defect in persistent_defects:
                # Color based on confidence (red->yellow->green)
                confidence_color = np.clip(defect.confidence / 10.0, 0, 1)
                color = (
                    0,
                    int(255 * (1 - confidence_color)),
                    int(255 * confidence_color)
                )
                cv2.circle(frame, (defect.x, defect.y), 2, color, -1)
                
            if out:
                out.write(frame)
                
            if display:
                cv2.imshow('Dead Pixel Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        self.logger.info("Processing complete")
        self.logger.info(f"Total persistent defects found: {len(self.persistent_defects)}")
        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

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
            output_path="output_video.mp4",
            display=True
        )
    except Exception as e:
        logging.error(f"Error processing video: {e}")

if __name__ == "__main__":
    main()
    