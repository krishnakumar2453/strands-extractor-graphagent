"""
For each image: read old full OCR → run validation → run word_candidates → save to image_output_v2.

Pipeline per image:
  1. Read image_output/{name}_full_ocr.txt (old full OCR)
  2. Run validate_ocr_from_image(image, full_ocr) → "NOTHING MISS" or corrected OCR
  3. Use original full_ocr if "NOTHING MISS", else use corrected text
  4. Run word_candidates.py on that text and save to image_output_v2/{name}_candidates.json and _pre_split.txt

Use --retry-failed to process only images that don't have image_output_v2/{name}_candidates.json yet
(skip already done; retry after 429 rate limit).

On 429 / quota errors, validation is retried with delay (parse retry_delay from error, default 30s, max 5 retries).
Later use compare_candidates.py to compare image_output vs image_output_v2 candidate lists.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from validate_ocr import validate_ocr_from_image

# Same layout as run_images_to_candidates / run_validate_ocr
IMAGES_DIR = Path("test_images")
OUTPUT_DIR = Path("image_output")
OUTPUT_V2 = Path("image_output_v2")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


def _find_image_for_base_name(base_name: str) -> Path | None:
    """Return path to image in IMAGES_DIR whose stem equals base_name, or None."""
    for ext in IMAGE_EXTENSIONS:
        p = IMAGES_DIR / f"{base_name}{ext}"
        if p.exists():
            return p
    return None


# Wait this many seconds before retrying after 429 rate limit
RATE_LIMIT_WAIT_SECONDS = 45


def _retry_wait_seconds(_error_message: str) -> int:
    """Seconds to wait before retrying after 429."""
    return RATE_LIMIT_WAIT_SECONDS


# Max validation retries on rate limit / transient errors
VALIDATE_MAX_RETRIES = 5


def main():
    parser = argparse.ArgumentParser(
        description="Validate OCR then run word_candidates; save to image_output_v2."
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Process only images that don't have image_output_v2/{name}_candidates.json yet (skip already done).",
    )
    args = parser.parse_args()

    if not OUTPUT_DIR.exists():
        print(f"Error: {OUTPUT_DIR} not found.")
        sys.exit(1)
    if not IMAGES_DIR.exists():
        print(f"Error: {IMAGES_DIR} not found.")
        sys.exit(1)

    OUTPUT_V2.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parent

    ocr_files = sorted(OUTPUT_DIR.glob("*_full_ocr.txt"))
    if not ocr_files:
        print(f"No *_full_ocr.txt files in {OUTPUT_DIR}")
        sys.exit(0)

    if args.retry_failed:
        ocr_files = [
            f for f in ocr_files
            if not (OUTPUT_V2 / f"{f.stem.replace('_full_ocr', '')}_candidates.json").exists()
        ]
        if not ocr_files:
            print("No failed/skipped images to process (all have v2 candidates).")
            sys.exit(0)
        print(f"Retrying {len(ocr_files)} image(s) without v2 output.\n")

    for i, ocr_path in enumerate(ocr_files, start=1):
        stem = ocr_path.stem
        base_name = stem[: -len("_full_ocr")] if stem.endswith("_full_ocr") else stem
        img_path = _find_image_for_base_name(base_name)
        if not img_path:
            print(f"[{i}] Skip (no image): {base_name}")
            continue

        full_ocr = ocr_path.read_text(encoding="utf-8")
        result = None
        for attempt in range(1, VALIDATE_MAX_RETRIES + 1):
            print(f"[{i}/{len(ocr_files)}] {base_name} validate (attempt {attempt}) ... ", end="", flush=True)
            result = validate_ocr_from_image(img_path, full_ocr)
            if result == "NOTHING MISS" or (not result.startswith("Error:")):
                break
            if "429" in result or "quota" in result.lower():
                wait = _retry_wait_seconds(result)
                print(f"429 → wait {wait}s ... ", end="", flush=True)
                time.sleep(wait)
            else:
                print(result)
                break
        else:
            print(result or "Error: max retries exceeded")
            continue

        if result == "NOTHING MISS":
            content = full_ocr
            status = "unchanged"
        else:
            content = result
            status = "corrected"

        ocr_v2_path = OUTPUT_V2 / f"{base_name}_full_ocr.txt"
        ocr_v2_path.write_text(content, encoding="utf-8")
        print(f"{status} → word_candidates ... ", end="", flush=True)
        subprocess.run(
            [sys.executable, "word_candidates.py", str(ocr_v2_path)],
            cwd=project_root,
            check=True,
        )
        print("ok")

    print("Done. Candidates and pre_split are in", OUTPUT_V2)


if __name__ == "__main__":
    main()
