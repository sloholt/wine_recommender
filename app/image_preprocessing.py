"""Step 1: clean a label photo so OCR and Claude get the same upright image.

Usage:
    python preprocess_label.py label.jpg
    python preprocess_label.py label.jpg --out-dir artifacts --rotation 90
    python preprocess_label.py label.jpg --crop 100 50 1900 2400 --preview

Requires Pillow:  pip install pillow
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps

MAX_SIZE = (2400, 2400)  # keeps small print (ABV, "750 ML") sharp for OCR
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def preprocess_image(
    image_path, output_path, crop_box=None, rotation=0, max_size=MAX_SIZE
):
    """Upright the photo, optionally crop/rotate, shrink, and save as JPEG.

    Vision and Claude both take an encoded image file. Pixel normalization to 0-1
    is only for models that expect numeric tensors, so it isn't done here.
    """
    with Image.open(image_path) as img:
        image = ImageOps.exif_transpose(img).convert(
            "RGB"
        )  # apply the phone's rotation flag
    if crop_box is not None:
        image = image.crop(crop_box)
    if rotation:
        image = image.rotate(rotation, expand=True, fillcolor="white")
    image.thumbnail(max_size)  # only shrinks, never enlarges
    image.save(output_path, format="JPEG", quality=95)
    return Path(output_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Clean a label photo for OCR and Claude."
    )
    parser.add_argument("image_path", type=Path, help="Path to the input label photo")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for the cleaned image (default: artifacts)",
    )
    parser.add_argument(
        "--output-name",
        default="clean_label.jpg",
        help="File name of the cleaned image (default: clean_label.jpg)",
    )
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="Optional crop box in pixels (experiments only)",
    )
    parser.add_argument(
        "--rotation",
        type=float,
        default=0,
        help="Degrees counter-clockwise (experiments only)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        nargs=2,
        default=list(MAX_SIZE),
        metavar=("WIDTH", "HEIGHT"),
        help="Maximum output size (default: 2400 2400)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Demo mode: skip preprocessing and use the original as-is",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Open a small preview of the cleaned image",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    image_path = args.image_path

    if not image_path.exists():
        print(f"Error: file not found: {image_path}", file=sys.stderr)
        return 1

    if not args.demo and image_path.suffix.lower() in IMAGE_TYPES:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        clean_image_path = preprocess_image(
            image_path,
            args.out_dir / args.output_name,
            crop_box=tuple(args.crop) if args.crop else None,
            rotation=args.rotation,
            max_size=tuple(args.max_size),
        )
        with Image.open(image_path) as original, Image.open(clean_image_path) as clean:
            print(f"Original: {original.size} -> Cleaned: {clean.size}")
            if args.preview:
                preview = clean.copy()
                preview.thumbnail((500, 500))  # small preview
                preview.show()
    else:
        clean_image_path = image_path
        print("Preprocessing skipped; using", image_path)

    print("Step 1 complete. Clean image:", clean_image_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
