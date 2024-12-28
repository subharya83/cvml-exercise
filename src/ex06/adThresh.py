import argparse 
import numpy as np
import cv2

def compute_integral_image(image):
    """
    Computes the integral image of a grayscale image.
    """
    return image.cumsum(axis=0).cumsum(axis=1)

def adaptive_thresholding(image, k, C):
    """
    Applies adaptive thresholding to a grayscale image using integral images.

    Parameters:
        image (np.ndarray): Grayscale input image.
        k (int): Size of the region to compute the mean (k x k).
        C (int): Constant to subtract from the mean for thresholding.

    Returns:
        np.ndarray: Binarized output image.
    """
    rows, cols = image.shape
    half_k = k // 2

    # Compute the integral image
    integral = compute_integral_image(image)

    # Output binary image
    binary_image = np.zeros_like(image, dtype=np.uint8)

    # Iterate through each pixel
    for i in range(rows):
        for j in range(cols):
            # Define the region boundaries
            y1 = max(0, i - half_k)
            y2 = min(rows - 1, i + half_k)
            x1 = max(0, j - half_k)
            x2 = min(cols - 1, j + half_k)

            # Compute the sum of pixel values in the region using the integral image
            region_sum = integral[y2, x2]
            if y1 > 0:
                region_sum -= integral[y1 - 1, x2]
            if x1 > 0:
                region_sum -= integral[y2, x1 - 1]
            if y1 > 0 and x1 > 0:
                region_sum += integral[y1 - 1, x1 - 1]

            # Compute the mean
            region_size = (y2 - y1 + 1) * (x2 - x1 + 1)
            region_mean = region_sum / region_size

            # Apply the threshold
            if image[i, j] < region_mean - C:
                binary_image[i, j] = 0  # Black
            else:
                binary_image[i, j] = 255  # White

    return binary_image

# Example usage
if __name__ == "__main__":
    # Load a grayscale image
    image = cv2.imread("example_image.jpg", cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error: Image not found.")
        exit()

    # Parameters
    k = 15  # Size of the region
    C = 10  # Constant to subtract from the mean

    # Apply adaptive thresholding
    binary_image = adaptive_thresholding(image, k, C)

    # Display the results
    cv2.imshow("Original Image", image)
    cv2.imshow("Binarized Image", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
