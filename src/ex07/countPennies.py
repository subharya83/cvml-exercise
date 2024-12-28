import argparse
import cv2
import numpy as np
import tensorflow as tf


def detect_and_count_pennies(image_path):
    """
    Detects and counts U.S. pennies in an image.
    
    Parameters:
        image_path (str): Path to the image containing coins.
        
    Returns:
        int: Number of pennies detected.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return 0

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=50
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        penny_count = 0

        for circle in circles[0, :]:
            x, y, radius = circle

            # Extract the region of interest (ROI) for the coin
            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), radius, 255, -1)
            coin_roi = cv2.bitwise_and(image, image, mask=mask)

            # Measure the diameter (pixel radius * 2)
            diameter = radius * 2

            # Pennies have a known approximate diameter
            if 18 <= diameter <= 20:  # Adjust thresholds based on calibration
                penny_count += 1

                # Optionally, draw the detected penny on the image
                cv2.circle(image, (x, y), radius, (0, 255, 0), 2)
                cv2.putText(image, "Penny", (x - radius, y - radius - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the result
        cv2.imshow("Detected Pennies", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return penny_count
    else:
        print("No coins detected.")
        return 0


def detect_and_count_pennies_with_cnn(image_path):
    """
    Detects and counts U.S. pennies in an image using a combination of Hough Circle Transform
    and a CNN-based object detection model.

    Parameters:
        image_path (str): Path to the image containing coins.
        
    Returns:
        int: Number of pennies detected.
    """
    model_path = "penny_detection_model.h5" 
    detection_model = tf.saved_model.load(model_path)

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return 0

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=50
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        penny_count = 0

        for circle in circles[0, :]:
            x, y, radius = circle

            # Extract the region of interest (ROI) for the coin
            x1, y1 = max(0, x - radius), max(0, y - radius)
            x2, y2 = min(image.shape[1], x + radius), min(image.shape[0], y + radius)
            coin_roi = image[y1:y2, x1:x2]

            # Prepare the ROI for CNN model
            coin_roi_resized = cv2.resize(coin_roi, (224, 224))  # Resize to match model input size
            coin_roi_tensor = tf.convert_to_tensor(coin_roi_resized, dtype=tf.float32)
            coin_roi_tensor = tf.expand_dims(coin_roi_tensor, axis=0)  # Add batch dimension

            # Perform inference with the CNN model
            detections = detection_model(coin_roi_tensor)

            # Extract detection results
            scores = detections["detection_scores"].numpy()
            classes = detections["detection_classes"].numpy()
            class_names = [int(cls) for cls in classes]

            # Check if the highest-scoring detection is a penny
            if scores[0] > 0.5 and class_names[0] == 1:  # Assuming class 1 is "Penny"
                penny_count += 1

                # Optionally, draw the detected penny on the image
                cv2.circle(image, (x, y), radius, (0, 255, 0), 2)
                cv2.putText(image, "Penny", (x - radius, y - radius - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the result
        cv2.imshow("Detected Pennies", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return penny_count
    else:
        print("No coins detected.")
        return 0

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Counting pennies w/o CNN')
    parser.add_argument('-i', required=True, help='Path to input image')
    # Parse arguments
    args = parser.parse_args()
    num_pennies = detect_and_count_pennies_with_cnn(args.i)
    print(f"Number of pennies detected: {num_pennies}")
