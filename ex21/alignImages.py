import argparse
import cv2
import numpy as np
from skimage import io, transform
from skimage.registration import phase_cross_correlation
from skimage.transform import warp, AffineTransform

def detect_and_match_keypoints(reference_image, target_image):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors in both images
    keypoints_ref, descriptors_ref = sift.detectAndCompute(reference_image, None)
    keypoints_target, descriptors_target = sift.detectAndCompute(target_image, None)

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

def align_images_sift(reference_image, target_image):
    # Detect keypoints and match them
    points_ref, points_target, good_matches = detect_and_match_keypoints(reference_image, target_image)

    # Compute homography using RANSAC
    homography, mask = cv2.findHomography(points_target, points_ref, cv2.RANSAC, 5.0)

    # Warp the target image using the homography
    height, width = reference_image.shape
    aligned_image = cv2.warpPerspective(target_image, homography, (width, height))

    # Extract translation parameters from homography
    tx, ty = extract_translation(homography)

    return aligned_image, tx, ty

def align_images_phase_correlation(reference_image, target_image):
    # Compute the translation parameters using phase correlation
    shift, error, diffphase = phase_cross_correlation(reference_image, target_image)

    # Apply the translation to the target image
    transform = AffineTransform(translation=(-shift[1], -shift[0]))
    aligned_image = warp(target_image, transform, mode='constant')

    return aligned_image, shift[1], shift[0]

def extract_translation(homography):
    # Normalize the homography matrix
    homography = homography / homography[2, 2]

    # Extract translation parameters
    tx = homography[0, 2]
    ty = homography[1, 2]

    return tx, ty

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Align a target image to a reference image.")
    parser.add_argument("-r", "--reference", required=True, help="Path to the reference image.")
    parser.add_argument("-t", "--target", required=True, help="Path to the target image.")
    parser.add_argument("-o", "--output", required=True, help="Path to save the aligned image.")
    parser.add_argument("-a", "--algorithm", type=int, required=True, choices=[0, 1],
                        help="Algorithm to use: 0 for SIFT, 1 for phase correlation.")
    args = parser.parse_args()

    # Load the reference and target images
    reference_image = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    target_image = cv2.imread(args.target, cv2.IMREAD_GRAYSCALE)

    # Check if images were loaded successfully
    if reference_image is None or target_image is None:
        print("Error: Could not load images.")
        return

    # Align the target image to the reference image using the selected algorithm
    if args.algorithm == 0:
        print("Using SIFT-based homography for alignment.")
        aligned_image, tx, ty = align_images_sift(reference_image, target_image)
    elif args.algorithm == 1:
        print("Using phase correlation for alignment.")
        aligned_image, tx, ty = align_images_phase_correlation(reference_image, target_image)
    else:
        print("Invalid algorithm selection.")
        return

    # Save the aligned image
    cv2.imwrite(args.output, aligned_image)

    # Print the translation parameters
    print(f"Translation parameters (in pixels): (tx, ty) = ({tx:.2f}, {ty:.2f})")

if __name__ == "__main__":
    main()