## Summed-Area-Tables(SAT)/Integral Images

### Problem Statement: Fast Image Smoothing with Adaptive Thresholding

You are developing a document scanner application that processes scanned images of text. One of the steps involves converting a grayscale image to a binary (black-and-white) image using **adaptive thresholding**, which ensures that text is clear and readable regardless of uneven lighting conditions.

#### Task
Write a program to binarize a grayscale image using the following approach:
1. Divide the image into overlapping regions of size \( k \times k \).
2. For each pixel, compute the mean intensity of the \( k \times k \) region centered on it.
3. Set the pixel to black if its intensity is less than the mean intensity minus a constant \( C \); otherwise, set it to white.

#### Constraints
- The image may be very large (e.g., 4K resolution), so a naive approach to compute the mean intensity for every pixel would be computationally expensive.
- The solution must run efficiently, making it suitable for real-time processing.

#### Input
- A grayscale image represented as a 2D array of pixel intensities.
- Parameters \( k \) (region size) and \( C \) (threshold adjustment constant).

#### Output
- A binarized image (2D array of binary values: 0 for black, 255 for white).

---

### Solution Using Integral Images

#### Steps to Solve:
1. Compute the **integral image** of the grayscale image.
   - This allows the sum of pixel intensities over any rectangular region to be computed in constant time.
2. For each pixel:
   - Use the integral image to compute the sum of pixel intensities in the \( k \times k \) region centered on the pixel.
   - Compute the mean intensity by dividing the sum by \( k^2 \).
   - Compare the pixel intensity to the threshold (\( \text{mean} - C \)) and set it accordingly.
3. Output the binarized image.

---

### Why Integral Images Are Useful
- The naive approach to compute the mean for every pixel requires \( O(k^2) \) operations per pixel, leading to \( O(N \times M \times k^2) \) for an image of size \( N \times M \).
- Using integral images, the sum of a region can be computed in \( O(1) \), reducing the complexity to \( O(N \times M) \).

This method ensures the binarization process is fast and scalable to high-resolution images.