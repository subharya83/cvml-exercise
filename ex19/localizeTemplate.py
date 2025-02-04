import numpy as np
import cv2
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass(order=True)
class SearchState:
    """Represents a search state for branch-and-bound algorithm"""
    priority: float = field(compare=True)
    t_range: Tuple[int, int] = field(compare=False)
    b_range: Tuple[int, int] = field(compare=False)
    l_range: Tuple[int, int] = field(compare=False)
    r_range: Tuple[int, int] = field(compare=False)

class EfficientSubwindowSearch:
    def __init__(self, target_image: np.ndarray, template_image: np.ndarray):
        """
        Initialize ESS template matching
        
        Args:
            target_image (np.ndarray): Full image to search in
            template_image (np.ndarray): Template to find
        """
        # Convert to grayscale
        self.target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        self.template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
        
        # Precompute integral images for fast bounds calculation
        self.target_integral = cv2.integral(self.target_gray)
        self.template_integral = cv2.integral(self.template_gray)
        
        # Template and target dimensions
        self.t_height, self.t_width = self.template_gray.shape
        self.img_height, self.img_width = self.target_gray.shape

    def compute_region_bounds(self, region):
        """
        Compute upper and lower bounds for a region
        
        Args:
            region (tuple): Region coordinates (top, bottom, left, right)
        
        Returns:
            tuple: Upper and lower bounds of the region
        """
        t, b, l, r = region
        
        # Compute region integral
        region_sum = (self.target_integral[b+1, r+1] + 
                      self.target_integral[t, l] - 
                      self.target_integral[t, r+1] - 
                      self.target_integral[b+1, l])
        
        # Compute total template sum
        template_sum = np.sum(self.template_gray)
        
        return region_sum, template_sum

    def normalized_correlation(self, region):
        """
        Compute normalized cross-correlation for a region
        
        Args:
            region (tuple): Region coordinates (top, bottom, left, right)
        
        Returns:
            float: Normalized cross-correlation score
        """
        t, b, l, r = region
        
        # Extract region
        region_img = self.target_gray[t:b+1, l:r+1]
        
        # Resize region to template size if needed
        if region_img.shape != self.template_gray.shape:
            region_img = cv2.resize(region_img, 
                                    (self.t_width, self.t_height), 
                                    interpolation=cv2.INTER_LINEAR)
        
        # Compute normalized cross-correlation
        correlation = np.sum((region_img - region_img.mean()) * 
                             (self.template_gray - self.template_gray.mean()))
        
        norm_1 = np.sqrt(np.sum((region_img - region_img.mean())**2))
        norm_2 = np.sqrt(np.sum((self.template_gray - self.template_gray.mean())**2))
        
        return correlation / (norm_1 * norm_2 + 1e-10)

    def bound_correlation(self, t_range, b_range, l_range, r_range):
        """
        Compute upper and lower bounds for correlation
        
        Args:
            t_range, b_range, l_range, r_range (tuple): Coordinate ranges
        
        Returns:
            tuple: Upper and lower correlation bounds
        """
        # Compute potential region bounds
        t_min, t_max = t_range
        b_min, b_max = b_range
        l_min, l_max = l_range
        r_min, r_max = r_range
        
        # Compute max and min possible regions
        max_region = (t_min, b_max, l_min, r_max)
        min_region = (t_max, b_min, l_max, r_min)
        
        return (self.normalized_correlation(max_region), 
                self.normalized_correlation(min_region))

    def efficient_subwindow_search(self, max_regions=1):
        """
        Perform efficient subwindow search
        
        Args:
            max_regions (int): Maximum number of regions to find
        
        Returns:
            list: Best matching regions [(top, bottom, left, right)]
        """
        # Initialize search queue
        search_queue = PriorityQueue()
        
        # Initial search state covers entire image
        initial_state = SearchState(
            priority=-1.0,  # Negative for max-heap behavior
            t_range=(0, self.img_height - self.t_height),
            b_range=(self.t_height - 1, self.img_height - 1),
            l_range=(0, self.img_width - self.t_width),
            r_range=(self.t_width - 1, self.img_width - 1)
        )
        
        search_queue.put(initial_state)
        best_regions = []
        
        while not search_queue.empty() and len(best_regions) < max_regions:
            current_state = search_queue.get()
            
            # If state is a single point, we've found a match
            if (current_state.t_range[0] == current_state.t_range[1] and
                current_state.b_range[0] == current_state.b_range[1] and
                current_state.l_range[0] == current_state.l_range[1] and
                current_state.r_range[0] == current_state.r_range[1]):
                
                best_regions.append((
                    current_state.t_range[0],
                    current_state.b_range[0],
                    current_state.l_range[0],
                    current_state.r_range[0]
                ))
                continue
            
            # Split state along largest coordinate range
            ranges = [
                current_state.t_range,
                current_state.b_range,
                current_state.l_range,
                current_state.r_range
            ]
            
            split_index = max(range(4), key=lambda i: ranges[i][1] - ranges[i][0])
            
            # Create two new states by splitting the range
            split_point = (ranges[split_index][0] + ranges[split_index][1]) // 2
            
            new_ranges = [list(r) for r in ranges]
            new_ranges[split_index][1] = split_point
            state1 = SearchState(
                priority=self.bound_correlation(*[tuple(r) for r in new_ranges])[0],
                t_range=tuple(new_ranges[0]),
                b_range=tuple(new_ranges[1]),
                l_range=tuple(new_ranges[2]),
                r_range=tuple(new_ranges[3])
            )
            
            new_ranges[split_index][0] = split_point + 1
            new_ranges[split_index][1] = ranges[split_index][1]
            state2 = SearchState(
                priority=self.bound_correlation(*[tuple(r) for r in new_ranges])[0],
                t_range=tuple(new_ranges[0]),
                b_range=tuple(new_ranges[1]),
                l_range=tuple(new_ranges[2]),
                r_range=tuple(new_ranges[3])
            )
            
            search_queue.put(state1)
            search_queue.put(state2)
        
        return best_regions

def locate_template_ess(target_image, template_image, max_regions=1):
    """
    Locate template in target image using Efficient Subwindow Search
    
    Args:
        target_image (np.ndarray): Full image to search in
        template_image (np.ndarray): Template to find
        max_regions (int): Maximum number of matching regions to find
    
    Returns:
        list: List of bounding boxes [(x, y, width, height)]
    """
    ess = EfficientSubwindowSearch(target_image, template_image)
    matches = ess.efficient_subwindow_search(max_regions)
    
    # Convert matches to (x, y, width, height) format
    bounding_boxes = []
    for match in matches:
        t, b, l, r = match
        bounding_boxes.append((l, t, r-l+1, b-t+1))
    
    return bounding_boxes

def main():
    # Load images
    target_image = cv2.imread('path/to/target_image.jpg')
    template_image = cv2.imread('path/to/template_image.jpg')
    
    # Find template locations
    matches = locate_template_ess(target_image, template_image)
    
    # Draw rectangles
    for (x, y, w, h) in matches:
        cv2.rectangle(target_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Show results
    cv2.imshow('ESS Template Matching', target_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()