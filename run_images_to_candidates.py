"""
Loop over all images in a folder: run OCR, save OCR as {name}_full_ocr.txt,
then run word_candidates to produce {name}_candidates.json and {name}_pre_split.txt.

Use --retry-failed to process only images that don't have output yet (skip already done).
"""
import argparse
import subprocess
import sys
from pathlib import Path

from ocr import tamil_ocr_from_image

# Config
IMAGES_DIR = Path("test_images")
OUTPUT_DIR = Path("image_output")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="OCR images and run word_candidates.")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Process only images that don't have output yet (skip if _full_ocr.txt exists).",
    )
    args = parser.parse_args()

    if not IMAGES_DIR.exists():
        print(f"Error: {IMAGES_DIR} not found.")
        sys.exit(1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(
        f for f in IMAGES_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        print(f"No images found in {IMAGES_DIR}")
        sys.exit(0)

    if args.retry_failed:
        images = [f for f in images if not (OUTPUT_DIR / f"{f.stem}_full_ocr.txt").exists()]
        if not images:
            print("No failed/skipped images to process (all have output).")
            sys.exit(0)
        print(f"Retrying {len(images)} image(s) without output.\n")

    for i, img_path in enumerate(images, start=1):
        name = img_path.stem
        print(f"[{i}/{len(images)}] {img_path.name} ...")
        # OCR
        text = tamil_ocr_from_image(img_path)
        if text.startswith("Error:"):
            print(f"  OCR failed: {text}")
            continue
        ocr_path = OUTPUT_DIR / f"{name}_full_ocr.txt"
        ocr_path.write_text(text, encoding="utf-8")
        print(f"  OCR saved: {ocr_path}")
        # Word candidates (same name → candidates + pre_split)
        subprocess.run(
            [sys.executable, "word_candidates.py", str(ocr_path)],
            cwd=Path(__file__).resolve().parent,
            check=True,
        )
    print("Done.")


if __name__ == "__main__":
    main()
