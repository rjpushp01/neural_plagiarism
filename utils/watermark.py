import os
import random
import cv2
import argparse
from imwatermark import WatermarkEncoder, WatermarkDecoder
from pathlib import Path
import sys

def generate_watermark(bits=32):
    """Generates a random 32-bit watermark."""
    random_int = random.getrandbits(bits)
    return random_int.to_bytes(int(bits/8), byteorder='big')

def watermark_image(image_path, watermark, method, target_folder, output_name=None):
    """Applies a watermark to an image using the specified method."""
    bgr = cv2.imread(str(image_path))

    # Initialize the encoder
    encoder = WatermarkEncoder()
    if method =='rivaGan':
        encoder.loadModel()

    encoder.set_watermark('bytes', watermark)

    # Encode the watermark using the specified method
    bgr_encoded = encoder.encode(bgr, method)

    # Save the watermarked image to the target folder
    if output_name is None:
        output_name = image_path.stem + "_watermarked" + image_path.suffix
    output_path = target_folder / output_name
    cv2.imwrite(str(output_path), bgr_encoded)

    print(f"Watermarked image saved to: {output_path}")
    return output_path

def save_watermark(watermark, target_folder, bits=32):
    """Saves the generated watermark to a binary file in the target folder."""
    watermark_file = target_folder / 'watermark_{}.bin'.format(bits)
    with open(watermark_file, 'wb') as f:
        f.write(watermark)
    print(f"Watermark saved to: {watermark_file}")

def process_images(image_folder, method, target_folder, bits):
    """Processes all images in the image folder and applies watermarking."""
    # Generate the watermark
    watermark = generate_watermark(bits)

    # Create target folder if it doesn't exist
    target_folder.mkdir(parents=True, exist_ok=True)

    # Save the watermark to the target folder
    save_watermark(watermark, target_folder)

    # Process each image in the folder
    for image_path in image_folder.glob('*'):
        if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
            watermark_image(image_path, watermark, method, target_folder)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Watermark images in a folder using a specified method.")
    parser.add_argument('--image_folder', type=str, help="Path to the folder containing images to watermark.")
    parser.add_argument('--method', type=str, help="Watermarking method to use (e.g., 'dwtDct', 'dct').")
    parser.add_argument('--target_folder', type=str, help="Path to the folder where watermarked images will be saved.")
    parser.add_argument('--bits', default=32, type=int, help="Watermark detection method to use (e.g., 'dwtDct', 'dct').")

    # Parse the arguments
    args = parser.parse_args()

    image_folder = Path(args.image_folder)
    method = args.method
    target_folder = Path(args.target_folder)
    bits = args.bits

    # Check if the image folder exists
    if not image_folder.exists() or not image_folder.is_dir():
        print(f"Error: The image folder '{image_folder}' does not exist or is not a directory.")
        sys.exit(1)

    # Process images
    process_images(image_folder, method, target_folder, bits)

if __name__ == "__main__":
    main()
