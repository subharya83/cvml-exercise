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

def detect_bright_regions(image, w, h, T):
    """
    Detects bright regions in a grayscale image using integral images.

    Parameters:
        image (np.ndarray): Grayscale input image.
        w (int): Width of the region.
        h (int): Height of the region.
        T (float): Intensity threshold.

    Returns:
        np.ndarray: An output image with bright regions highlighted.
    """
    rows, cols = image.shape
    half_w = w // 2
    half_h = h // 2

    # Compute the integral image
    integral = compute_integral_image(image)

    # Output image to highlight bright regions
    output_image = np.zeros_like(image, dtype=np.uint8)

    # Iterate through each pixel
    for i in range(rows):
        for j in range(cols):
            # Define the region boundaries
            y1 = max(0, i - half_h)
            y2 = min(rows - 1, i + half_h)
            x1 = max(0, j - half_w)
            x2 = min(cols - 1, j + half_w)

            # Compute the sum of pixel values in the region using the integral image
            region_sum = integral[y2, x2]
            if y1 > 0:
                region_sum -= integral[y1 - 1, x2]
            if x1 > 0:
                region_sum -= integral[y2, x1 - 1]
            if y1 > 0 and x1 > 0:
                region_sum += integral[y1 - 1, x1 - 1]

            # Compute the mean intensity of the region
            region_size = (y2 - y1 + 1) * (x2 - x1 + 1)
            region_mean = region_sum / region_size

            # Highlight the region if the mean intensity exceeds the threshold
            if region_mean > T:
                output_image[i, j] = 255  # Highlight

    return output_image

# Example usage
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Integral image practical usecases')
    parser.add_argument('-i', required=True, help='Path to input image')
    
    parser.add_argument('-o', required=True, help='Path to output image')
    
    # Parse arguments
    args = parser.parse_args()
  
    # Load a grayscale image
    image = cv2.imread(args.i, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error: Image not found.")
        exit()

    if args.m == 0:
        # Parameters
        k = 15  # Size of the region
        C = 10  # Constant to subtract from the mean

        # Apply adaptive thresholding
        output_image = adaptive_thresholding(image, k, C)
    elif args.m == 1: 
        # Parameters
        w = 50  # Width of the region
        h = 50  # Height of the region
        T = 200  # Intensity threshold

        # Detect bright regions
        output_image = detect_bright_regions(image, w, h, T)

    cv2.imwrite(args.o, output_image)
    # Display the results
    cv2.imshow("Original Image", image)
    cv2.imshow("Output Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
