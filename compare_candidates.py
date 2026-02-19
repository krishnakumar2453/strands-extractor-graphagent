"""
Compare word candidate lists: image_output (v1, from original/full OCR) vs image_output_v2 (from validated OCR when available).

Loads image_output/{name}_candidates.json and image_output_v2/{name}_candidates.json for each name
and reports same / different, and per-image diffs (only in v1, only in v2).
"""
import json
import sys
from pathlib import Path

OUTPUT_DIR = Path("image_output")
OUTPUT_V2 = Path("image_output_v2")


def main():
    if not OUTPUT_DIR.exists():
        print(f"Error: {OUTPUT_DIR} not found.")
        sys.exit(1)
    if not OUTPUT_V2.exists():
        print(f"Error: {OUTPUT_V2} not found. Run run_validated_to_candidates.py first.")
        sys.exit(1)

    v1_files = sorted(OUTPUT_DIR.glob("*_candidates.json"))
    if not v1_files:
        print(f"No *_candidates.json in {OUTPUT_DIR}")
        sys.exit(0)

    same_count = 0
    diff_count = 0
    only_v2_count = 0  # in v2 but no v1

    for v1_path in v1_files:
        base_name = v1_path.stem.replace("_candidates", "")
        v2_path = OUTPUT_V2 / f"{base_name}_candidates.json"
        if not v2_path.exists():
            print(f"[skip] {base_name}: no v2 file")
            continue

        v1_list = json.loads(v1_path.read_text(encoding="utf-8"))
        v2_list = json.loads(v2_path.read_text(encoding="utf-8"))
        v1_set = set(v1_list)
        v2_set = set(v2_list)

        if v1_list == v2_list:
            same_count += 1
            print(f"[same] {base_name}: {len(v1_list)} candidates")
            continue

        diff_count += 1
        only_in_v1 = v1_set - v2_set
        only_in_v2 = v2_set - v1_set
        print(f"[diff] {base_name}: v1={len(v1_list)}, v2={len(v2_list)}")
        if only_in_v1:
            print(f"       only in v1 ({len(only_in_v1)}): {sorted(only_in_v1)[:10]}{' ...' if len(only_in_v1) > 10 else ''}")
        if only_in_v2:
            print(f"       only in v2 ({len(only_in_v2)}): {sorted(only_in_v2)[:10]}{' ...' if len(only_in_v2) > 10 else ''}")

    print()
    print(f"Summary: same={same_count}, different={diff_count}")


if __name__ == "__main__":
    main()
