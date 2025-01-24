## Outlier Detection Using Summed-Area Tables (SAT) / Integral Images

Welcome to an exciting exploration of how **Summed-Area Tables (SAT)**, also known as **Integral Images**, can be leveraged for efficient outlier detection in images. This project is designed to challenge and inspire Computer Vision graduate students by diving into two practical problems: 
**Adaptive Thresholding** for document scanning and **Bright Region Detection** in satellite imagery. Both tasks are implemented in both C++ and 
Python, providing a comprehensive understanding of the techniques and their real-world applications.

### Problem 1: Fast Image Smoothing with Adaptive Thresholding

Imagine you're developing a document scanner application that processes scanned images of text. One of the critical steps is converting a grayscale image to a binary (black-and-white) image using **adaptive thresholding**. This ensures that text remains clear and readable, even under uneven lighting conditions.

#### Task
Your goal is to binarize a grayscale image using the following approach:
1. **Divide the image** into overlapping regions of size \( k \times k \).
2. **For each pixel**, compute the mean intensity of the \( k \times k \) region centered on it.
3. **Set the pixel** to black if its intensity is less than the mean intensity minus a constant \( C \); otherwise, set it to white.

#### Constraints
- The image may be very large (e.g., 4K resolution), so a naive approach to compute the mean intensity for every pixel would be computationally expensive.
- The solution must run efficiently, making it suitable for real-time processing.

#### Input
- A grayscale image represented as a 2D array of pixel intensities.
- Parameters \( k \) (region size) and \( C \) (threshold adjustment constant).

#### Output
- A binarized image (2D array of binary values: 0 for black, 255 for white).

### Problem 2: Fast Detection of Bright Regions in Satellite Images

Now, let's shift our focus to analyzing high-resolution satellite images. The task is to identify regions with high reflectance, such as snow-covered areas, water bodies, or urban regions with bright rooftops. The goal is to locate and highlight all areas in the image where the average pixel intensity within a given rectangular region exceeds a certain threshold.

#### Task
Write a program that:
1. **Divides the satellite image** into overlapping rectangular regions of size \( w \times h \).
2. **Calculates the average intensity** of each region.
3. **Highlights all regions** where the average intensity exceeds a specified threshold \( T \).

#### Constraints
- The satellite images are large (e.g., 8K resolution), so the solution must be efficient.
- The regions may overlap, and the calculations for each region should be performed quickly.

#### Input
- A grayscale satellite image represented as a 2D array of pixel intensities.
- Parameters \( w \) (width of the region), \( h \) (height of the region), and \( T \) (intensity threshold).

#### Output
- An image where the bright regions are highlighted, with all other regions darkened.

### Practical Applications
- **Environmental Monitoring**: Detecting snow-covered areas, drought-affected regions, or water bodies.
- **Urban Planning**: Identifying areas with reflective rooftops or high urban density.
- **Agriculture**: Monitoring crop health through reflectance patterns.

### Solution Using Integral Images

#### Steps to Solve:
1. **Compute the integral image** of the grayscale image.
   - This allows the sum of pixel intensities over any rectangular region to be computed in constant time.
2. **For each pixel**:
   - Use the integral image to compute the sum of pixel intensities in the \( k \times k \) region centered on the pixel.
   - Compute the mean intensity by dividing the sum by \( k^2 \).
   - Compare the pixel intensity to the threshold (\( \text{mean} - C \)) and set it accordingly.
3. **Output the binarized image**.

---

### Why Integral Images Are Useful
- The **naive approach** to compute the mean for every pixel requires \( O(k^2) \) operations per pixel, leading to \( O(N \times M \times k^2) \) for an image of size \( N \times M \).
- Using **integral images**, the sum of a region can be computed in \( O(1) \), reducing the complexity to \( O(N \times M) \).

This method ensures the binarization process is fast and scalable to high-resolution images.

### Implementation Details

#### C++ Implementation
The C++ implementation (`adThresh.cpp`) provides a robust and efficient solution using OpenCV. It includes functions for:
- **Computing the integral image**.
- **Adaptive thresholding** using integral images.
- **Detecting bright regions** in satellite images.

#### Python Implementation
The Python implementation (`adThresh.py`) mirrors the C++ functionality, offering a more accessible and flexible approach for rapid prototyping and experimentation. It uses OpenCV and NumPy for efficient image processing.

### How to Run the Code
1. **C++**:
   - Compile the code using a C++ compiler with OpenCV installed.
   - Run the executable with the appropriate command-line arguments:
     ```bash
     ./adThresh -i <input_image> -m <mode> [-o <output_image>]
     ```
   - Modes:
     - `0` for adaptive thresholding.
     - `1` for bright region detection.

2. **Python**:
   - Ensure you have Python installed with OpenCV and NumPy.
   - Run the script with the required arguments:
     ```bash
     python adThresh.py -i <input_image> -m <mode> [-o <output_image>]
     ```
   - Modes:
     - `0` for adaptive thresholding.
     - `1` for bright region detection.

### Example Outputs
- **Adaptive Thresholding**: Converts a grayscale image of text into a clean, binary image, enhancing readability.
- **Bright Region Detection**: Highlights areas in a satellite image where the average intensity exceeds the threshold, useful for identifying snow, water, or urban areas.

### Challenges for Graduate Students
- **Optimization**: Explore further optimizations to reduce computation time, especially for ultra-high-resolution images.
- **Parameter Tuning**: Experiment with different values of \( k \), \( C \), \( w \), \( h \), and \( T \) to understand their impact on the results.
- **Extension**: Extend the code to handle color images or to detect other types of regions (e.g., dark regions).

### Conclusion
This project offers a hands-on experience with integral images, a powerful tool in computer vision. By solving real-world problems like document scanning and satellite image analysis, you'll gain valuable insights into efficient image processing techniques. Whether you're optimizing for speed or exploring new applications, this project is a stepping stone to mastering advanced computer vision algorithms.
