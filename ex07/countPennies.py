import argparse
import cv2
import numpy as np

def detect_circles(image_path, visualize=False):
    """
    Detects circular objects in an image using Hough Circle Transform.

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

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,  # Inverse ratio of the accumulator resolution
        minDist=30,  # Minimum distance between detected centers
        param1=50,  # Upper threshold for edge detection
        param2=30,  # Threshold for center detection
        minRadius=10,  # Minimum radius of circles to detect
        maxRadius=500   # Maximum radius of circles to detect
    )

    if circles is not None:
        # Convert the circle parameters to integers
        circles = np.uint16(np.around(circles))
        num_circles = len(circles[0, :])

        if visualize:
            # Draw the detected circles on the original image
            for circle in circles[0, :]:
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
        print("No circular objects detected.")
        return 0


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Detect circular objects in an image.")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image.")
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize the detected circles on the output image.")

    # Parse arguments
    args = parser.parse_args()

    # Detect circles in the image
    num_circles = detect_circles(args.image, visualize=args.visualize)

    # Print the number of detected circles
    print(f"Number of circular objects detected: {num_circles}")