import argparse
import cv2
import numpy as np

def detect_and_match_keypoints(reference_image, target_image):
    # Initialize SURF detector
    surf = cv2.SIFT_create()  # SIFT is used as SURF is not available in default OpenCV installations

    # Detect keypoints and descriptors in both images
    keypoints_ref, descriptors_ref = surf.detectAndCompute(reference_image, None)
    keypoints_target, descriptors_target = surf.detectAndCompute(target_image, None)

    # FLANN-based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors
    matches = flann.knnMatch(descriptors_ref, descriptors_target, k=2)

    # Filter matches using the Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Extract location of good matches
    points_ref = np.float32([keypoints_ref[m.queryIdx].pt for m in good_matches])
    points_target = np.float32([keypoints_target[m.trainIdx].pt for m in good_matches])

    return points_ref, points_target, good_matches

def align_images(reference_image, target_image):
    # Detect keypoints and match them
    points_ref, points_target, good_matches = detect_and_match_keypoints(reference_image, target_image)

    # Compute homography using RANSAC
    homography, mask = cv2.findHomography(points_target, points_ref, cv2.RANSAC, 5.0)

    # Warp the target image using the homography
    height, width = reference_image.shape
    aligned_image = cv2.warpPerspective(target_image, homography, (width, height))

    return aligned_image, homography

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Align a target image to a reference image using homography.")
    parser.add_argument("-r", "--reference", required=True, help="Path to the reference image.")
    parser.add_argument("-t", "--target", required=True, help="Path to the target image.")
    parser.add_argument("-o", "--output", required=True, help="Path to save the aligned image.")
    args = parser.parse_args()

    # Load the reference and target images
    reference_image = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    target_image = cv2.imread(args.target, cv2.IMREAD_GRAYSCALE)

    # Check if images were loaded successfully
    if reference_image is None or target_image is None:
        print("Error: Could not load images.")
        return

    # Align the target image to the reference image
    aligned_image, homography = align_images(reference_image, target_image)

    # Save the aligned image
    cv2.imwrite(args.output, aligned_image)

    # Print the homography matrix
    print("Homography matrix:")
    print(homography)

if __name__ == "__main__":
    main()
    