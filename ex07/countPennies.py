import argparse
import cv2
import numpy as np

def non_max_suppression(circles, overlap_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to filter out overlapping circles.

    Parameters:
        circles (np.array): Array of detected circles in the format (x, y, radius).
        overlap_threshold (float): Threshold for overlapping area ratio to suppress a circle.

    Returns:
        np.array: Filtered circles after NMS.
    """
    if circles is None or len(circles) == 0:
        return np.array([])

    # Convert circles to a list of (x, y, radius) tuples
    circles = np.uint16(np.around(circles))
    circles = circles[0, :]  # Remove the extra dimension
    circles = sorted(circles, key=lambda x: x[2], reverse=True)  # Sort by radius (largest first)

    suppressed = []

    while len(circles) > 0:
        # Take the largest circle
        current = circles[0]
        suppressed.append(current)
        circles = circles[1:]

        # Calculate overlap with remaining circles
        to_remove = []
        for i, circle in enumerate(circles):
            # Calculate distance between centers
            dx = current[0] - circle[0]
            dy = current[1] - circle[1]
            distance = np.sqrt(dx**2 + dy**2)

            # Calculate overlap ratio (intersection over union)
            if distance < (current[2] + circle[2]):
                # Circles overlap
                overlap_ratio = (current[2] ** 2) / (circle[2] ** 2)
                if overlap_ratio > overlap_threshold:
                    to_remove.append(i)

        # Remove suppressed circles
        circles = [circle for i, circle in enumerate(circles) if i not in to_remove]

    return np.array([suppressed])


def detect_circles(image_path, visualize=False):
    """
    Detects circular objects in an image using Hough Circle Transform and applies NMS.

    Parameters:
        image_path (str): Path to the input image.
        visualize (bool): Whether to visualize the detected circles on the output image.

    Returns:
        int: Number of circular objects detected.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return 0

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    edges = cv2.Canny(blurred, 50, 150)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,  # Inverse ratio of the accumulator resolution
        minDist=30,  # Minimum distance between detected centers
        param1=50,  # Upper threshold for edge detection
        param2=30,  # Threshold for center detection
        minRadius=10,  # Minimum radius of circles to detect
        maxRadius=500   # Maximum radius of circles to detect
    )

    if circles is not None:
        # Apply Non-Maximum Suppression (NMS)
        filtered_circles = non_max_suppression(circles, overlap_threshold=0.5)

        if len(filtered_circles) > 0:
            num_circles = len(filtered_circles[0, :])

            if visualize:
                # Draw the detected circles on the original image
                for circle in filtered_circles[0, :]:
                    x, y, radius = circle
                    # Draw the outer circle
                    cv2.circle(image, (x, y), radius, (0, 255, 0), 2)
                    # Draw the center of the circle
                    cv2.circle(image, (x, y), 2, (0, 0, 255), 3)

                # Show the output image with detected circles
                cv2.imshow("Detected Circles", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return num_circles
        else:
            print("No circular objects detected after NMS.")
            return 0
    else:
        print("No circular objects detected.")
        return 0


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Detect circular objects in an image with NMS.")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image.")
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize the detected circles on the output image.")

    # Parse arguments
    args = parser.parse_args()

    # Detect circles in the image
    num_circles = detect_circles(args.image, visualize=args.visualize)

    # Print the number of detected circles
    print(f"Number of circular objects detected: {num_circles}")