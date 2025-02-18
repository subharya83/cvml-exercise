import argparse
import numpy as np
from skimage import io, transform
from skimage.registration import phase_cross_correlation
from skimage.transform import warp, AffineTransform

def align_images(reference_image, target_image):
    # Compute the translation parameters using phase correlation
    shift, error, diffphase = phase_cross_correlation(reference_image, target_image)

    # Apply the translation to the target image
    transform = AffineTransform(translation=(-shift[1], -shift[0]))
    aligned_image = warp(target_image, transform, mode='constant')

    return aligned_image, shift

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Align a target image to a reference image.")
    parser.add_argument("-r", "--reference", required=True, help="Path to the reference image.")
    parser.add_argument("-t", "--target", required=True, help="Path to the target image.")
    parser.add_argument("-o", "--output", required=True, help="Path to save the aligned image.")
    args = parser.parse_args()

    # Load the reference and target images
    reference_image = io.imread(args.reference, as_gray=True)
    target_image = io.imread(args.target, as_gray=True)

    # Align the target image to the reference image
    aligned_image, shift = align_images(reference_image, target_image)

    # Save the aligned image
    io.imsave(args.output, aligned_image)

    # Print the translation parameters
    print(f"Translation parameters (in pixels): (x, y) = ({shift[1]}, {shift[0]})")

if __name__ == "__main__":
    main()