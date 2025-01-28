import argparse
import cv2
import numpy as np
import tensorflow as tf


def detect_and_count_pennies(image_path):
    """
    Detects and counts U.S. pennies in an image using multi-scale Hough Circle Transform.
    
    Parameters:
        image_path (str): Path to the image containing coins.
        
    Returns:
        int: Number of pennies detected.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Multi-scale parameters
    scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    all_circles = []

    for s in scales:
        resized = cv2.resize(blurred, None, fx=s, fy=s)
        min_radius_scaled = max(1, int(10 * s))  # Ensure min radius is at least 1
        max_radius_scaled = int(50 * s)
        
        circles = cv2.HoughCircles(
            resized,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=min_radius_scaled,
            maxRadius=max_radius_scaled
        )
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                # Convert back to original scale
                orig_x = int(x / s)
                orig_y = int(y / s)
                orig_r = int(r / s)
                all_circles.append((orig_x, orig_y, orig_r))

    # Merge overlapping circles
    merged = []
    for x, y, r in all_circles:
        duplicate = False
        for mx, my, mr in merged:
            distance = np.sqrt((x - mx)**2 + (y - my)**2)
            if distance < 20 and abs(r - mr) < 10:
                duplicate = True
                break
        if not duplicate:
            merged.append((x, y, r))

    penny_count = 0
    for x, y, r in merged:
        diameter = 2 * r
        # Adjusted diameter range based on Hough parameters and calibration
        if 18 <= diameter <= 22:  # Calibrate this range as needed
            penny_count += 1
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            cv2.putText(image, "Penny", (x - r, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Detected Pennies", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return penny_count


def detect_and_count_pennies_with_cnn(image_path):
    """
    Detects and counts pennies using multi-scale Hough Transform and CNN classification.
    """
    model_path = "penny_detection_model.h5"
    detection_model = tf.saved_model.load(model_path)

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    all_circles = []

    for s in scales:
        resized = cv2.resize(blurred, None, fx=s, fy=s)
        min_radius_scaled = max(1, int(10 * s))
        max_radius_scaled = int(1000 * s)
        
        circles = cv2.HoughCircles(
            resized,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=min_radius_scaled,
            maxRadius=max_radius_scaled
        )
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                orig_x = int(x / s)
                orig_y = int(y / s)
                orig_r = int(r / s)
                all_circles.append((orig_x, orig_y, orig_r))

    merged = []
    for x, y, r in all_circles:
        duplicate = False
        for mx, my, mr in merged:
            distance = np.sqrt((x - mx)**2 + (y - my)**2)
            if distance < 20 and abs(r - mr) < 10:
                duplicate = True
                break
        if not duplicate:
            merged.append((x, y, r))

    penny_count = 0
    for x, y, r in merged:
        x1, y1 = max(0, x - r), max(0, y - r)
        x2, y2 = x + r, y + r
        coin_roi = image[y1:y2, x1:x2]
        if coin_roi.size == 0:
            continue

        coin_roi_resized = cv2.resize(coin_roi, (224, 224))
        coin_roi_tensor = tf.convert_to_tensor(coin_roi_resized, dtype=tf.float32)
        coin_roi_tensor = tf.expand_dims(coin_roi_tensor, axis=0)

        detections = detection_model(coin_roi_tensor)
        scores = detections["detection_scores"].numpy()[0]
        classes = detections["detection_classes"].numpy()[0]

        for score, cls in zip(scores, classes):
            if score > 0.5 and cls == 1:
                penny_count += 1
                cv2.circle(image, (x, y), r, (0, 255, 0), 2)
                cv2.putText(image, "Penny", (x - r, y - r - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                break

    cv2.imshow("Detected Pennies", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return penny_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count pennies using improved detection')
    parser.add_argument('-i', required=True, help='Path to input image')
    parser.add_argument('--cnn', action='store_true', help='Use CNN for classification')
    args = parser.parse_args()
    
    if args.cnn:
        num_pennies = detect_and_count_pennies_with_cnn(args.i)
    else:
        num_pennies = detect_and_count_pennies(args.i)
        
    print(f"Number of pennies detected: {num_pennies}")