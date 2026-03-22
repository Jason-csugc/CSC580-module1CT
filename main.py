import argparse
import face_recognition
import numpy as np
from PIL import Image, ImageDraw

import os

def main():
    parser = argparse.ArgumentParser(description='Detect and draw faces in an image')
    parser.add_argument('image_path', nargs='?', default='crowd.jpg',
                        help='Path to the input image file (default: crowd.jpg)')
    parser.add_argument('--output', '-o', default='image_with_boxes.jpg',
                        help='Path to save the output image with boxes (default: image_with_boxes.jpg)')
    parser.add_argument('--positions', '-p', action='store_true',
                        help='Print the positions of detected faces to the console'
                        ' (format: Top, Right, Bottom, Left)')
    parser.add_argument('--show', '-s', action='store_true',
                        help='Show the result image after processing')
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        print(f"Error: input file does not exist: {args.image_path}")
        raise SystemExit(1)

    try:
        image = Image.open(args.image_path)
    except Exception as exc:
        print(f"Error: failed to open image '{args.image_path}': {exc}")
        raise SystemExit(1) from exc

    img_array = np.array(image)

    # Convert image to RGB if needed (face_recognition works with RGB)
    if image.mode != 'RGB':
        image = image.convert('RGB')
        img_array = np.array(image)

    # Find all faces in the image
    face_locations = face_recognition.face_locations(img_array)

    # Draw red boxes around each face
    draw = ImageDraw.Draw(image)
    for top, right, bottom, left in face_locations:
        draw.rectangle([left, top, right, bottom], outline='red', width=3)

    # Determine output filename when not explicitly provided
    if parser.get_default('output') == args.output:
        base, ext = os.path.splitext(args.image_path)
        if not ext:
            ext = '.jpg'
        args.output = f"{base}_with_boxes{ext}"

    # Save the image with boxes
    image.save(args.output)
    print(f"Image saved as '{args.output}' with {len(face_locations)} face(s) highlighted")

    # Print results
    if args.positions:
        print(f"Found {len(face_locations)} face(s) in the image: {args.image_path}")
        for i, (top, right, bottom, left) in enumerate(face_locations, 1):
            print(f"Face {i}: Top={top}, Right={right}, Bottom={bottom}, Left={left}")

    # Display the image on the screen if requested
    if args.show:
        image.show()


if __name__ == '__main__':
    main()
