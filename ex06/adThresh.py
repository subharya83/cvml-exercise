import cv2
import numpy as np
import sys
import argparse

def compute_integral_image(image):
    integral = cv2.integral(image, cv2.CV_32S)
    return integral

def adaptive_thresholding(image, k, C):
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
            region_sum = integral[y2 + 1, x2 + 1]
            if y1 > 0:
                region_sum -= integral[y1, x2 + 1]
            if x1 > 0:
                region_sum -= integral[y2 + 1, x1]
            if y1 > 0 and x1 > 0:
                region_sum += integral[y1, x1]

            # Compute the mean
            region_size = (y2 - y1 + 1) * (x2 - x1 + 1)
            region_mean = region_sum // region_size

            # Apply the threshold
            if image[i, j] < region_mean - C:
                binary_image[i, j] = 0  # Black
            else:
                binary_image[i, j] = 255  # White

    return binary_image

def detect_bright_regions(image, w, h, T):
    rows, cols = image.shape
    half_w = w // 2
    half_h = h // 2

    # Compute the integral image
    integral = compute_integral_image(image)

    # Output image to highlight bright regions
    output_image = np.zeros_like(image, dtype=np.uint8)

    # To store coordinates of bright regions
    bright_regions = []

    # Iterate through each pixel
    for i in range(rows):
        for j in range(cols):
            # Define the region boundaries
            y1 = max(0, i - half_h)
            y2 = min(rows - 1, i + half_h)
            x1 = max(0, j - half_w)
            x2 = min(cols - 1, j + half_w)

            # Compute the sum of pixel values in the region using the integral image
            region_sum = integral[y2 + 1, x2 + 1]
            if y1 > 0:
                region_sum -= integral[y1, x2 + 1]
            if x1 > 0:
                region_sum -= integral[y2 + 1, x1]
            if y1 > 0 and x1 > 0:
                region_sum += integral[y1, x1]

            # Compute the mean intensity of the region
            region_size = (y2 - y1 + 1) * (x2 - x1 + 1)
            region_mean = region_sum / region_size

            # Highlight the region if the mean intensity exceeds the threshold
            if region_mean > T:
                output_image[i, j] = 255  # Highlight
                bright_regions.append((j, i))  # Store the coordinates of the bright region

    return output_image, bright_regions

def main():
    parser = argparse.ArgumentParser(description="Process an image using adaptive thresholding or bright region detection.")
    parser.add_argument("-i", required=True, help="Path to the input image")
    parser.add_argument("-m", type=int, required=True, help="Mode: 0 for adaptive thresholding, 1 for bright region detection")
    parser.add_argument("-o", help="Path to save the output image")
    args = parser.parse_args()

    # Load the grayscale image
    image = cv2.imread(args.i, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        return -1

    output_image = None
    bright_regions = []  # To store coordinates of bright regions

    if args.m == 0:
        # Parameters for adaptive thresholding
        k = 15  # Size of the region
        C = 10  # Constant to subtract from the mean

        # Apply adaptive thresholding
        output_image = adaptive_thresholding(image, k, C)
    elif args.m == 1:
        # Parameters for detecting bright regions
        w = 50  # Width of the region
        h = 50  # Height of the region
        T = 200.0  # Intensity threshold

        # Detect bright regions
        output_image, bright_regions = detect_bright_regions(image, w, h, T)

        # Output the number of detected bright regions and their coordinates
        print("Number of bright regions detected:", len(bright_regions))
        print("Coordinates of bright regions (x, y):")
        for point in bright_regions:
            print(f"({point[0]}, {point[1]})")
    else:
        print("Invalid mode selected. Use 0 for adaptive thresholding or 1 for bright region detection.")
        return -1

    # Save the output image only if -o is provided
    if args.o:
        cv2.imwrite(args.o, output_image)
        print("Output image saved to:", args.o)

if __name__ == "__main__":
    main()