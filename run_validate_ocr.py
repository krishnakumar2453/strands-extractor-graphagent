"""
Optional step: validate existing OCR in image_output against the source images.
Reads each {name}_full_ocr.txt, runs validate_ocr_from_image(image, ocr_text).
If nothing is missing: prints "NOTHING MISS" for that image.
If something is missing: writes {name}_validated_ocr.txt with the corrected OCR.

Use --skip-done to skip images that already have _validated_ocr.txt.

Single image: pass the image path or base name (e.g. "test_image (37)" or "test_images/test_image (37).png").
"""
import argparse
import sys
from pathlib import Path

from validate_ocr import validate_ocr_from_image

# Match run_images_to_candidates layout
IMAGES_DIR = Path("test_images")
OUTPUT_DIR = Path("image_output")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


def _find_image_for_base_name(base_name: str) -> Path | None:
    """Return path to image in IMAGES_DIR whose stem equals base_name, or None."""
    for ext in IMAGE_EXTENSIONS:
        p = IMAGES_DIR / f"{base_name}{ext}"
        if p.exists():
            return p
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Validate OCR text against images; write corrected OCR or report NOTHING MISS."
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Optional: validate only this image (path or base name, e.g. 'test_image (37)' or 'test_images/test_image (37).png').",
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="Skip images that already have _validated_ocr.txt.",
    )
    args = parser.parse_args()

    if not OUTPUT_DIR.exists():
        print(f"Error: {OUTPUT_DIR} not found.")
        sys.exit(1)
    if not IMAGES_DIR.exists():
        print(f"Error: {IMAGES_DIR} not found.")
        sys.exit(1)

    if args.image:
        # Single image: resolve to (img_path, ocr_path)
        raw = Path(args.image)
        if raw.exists():
            img_path = raw.resolve()
            base_name = img_path.stem
        else:
            base_name = raw.stem if raw.suffix.lower() in IMAGE_EXTENSIONS else args.image.strip()
            img_path = _find_image_for_base_name(base_name)
            if not img_path:
                print(f"Error: No image found for '{args.image}' in {IMAGES_DIR}")
                sys.exit(1)
        ocr_path = OUTPUT_DIR / f"{base_name}_full_ocr.txt"
        if not ocr_path.exists():
            print(f"Error: No OCR file {ocr_path.name}")
            sys.exit(1)
        ocr_files = [ocr_path]
        if args.skip_done and (OUTPUT_DIR / f"{base_name}_validated_ocr.txt").exists():
            print(f"Already has _validated_ocr.txt: {base_name}")
            sys.exit(0)
    else:
        ocr_files = sorted(OUTPUT_DIR.glob("*_full_ocr.txt"))
        if not ocr_files:
            print(f"No *_full_ocr.txt files in {OUTPUT_DIR}")
            sys.exit(0)
        if args.skip_done:
            ocr_files = [
                f for f in ocr_files
                if not (OUTPUT_DIR / f"{f.stem.replace('_full_ocr', '')}_validated_ocr.txt").exists()
            ]
            if not ocr_files:
                print("All images already have _validated_ocr.txt.")
                sys.exit(0)

    for i, ocr_path in enumerate(ocr_files, start=1):
        stem = ocr_path.stem
        base_name = stem[: -len("_full_ocr")] if stem.endswith("_full_ocr") else stem
        img_path = _find_image_for_base_name(base_name)
        if not img_path:
            print(f"[{i}] Skip (no image): {base_name}")
            continue

        ocr_text = ocr_path.read_text(encoding="utf-8")
        print(f"[{i}] Validating: {img_path.name} ... ", end="", flush=True)
        result = validate_ocr_from_image(img_path, ocr_text)

        if result == "NOTHING MISS":
            print("NOTHING MISS")
        elif result.startswith("Error:"):
            print(result)
        else:
            out_path = OUTPUT_DIR / f"{base_name}_validated_ocr.txt"
            out_path.write_text(result, encoding="utf-8")
            print(f"corrected → {out_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
