"""
One-time baseline capture: run main.py on each PDF in test_pdfs/ and copy
the resulting *_vocabulary.json into testcases/<basename>/root_output.json.
"""
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_PDF_DIR = Path("test_pdfs")
DEFAULT_TESTCASES_DIR = Path("testcases")
RUN_OUTPUT_DIR = Path("pdf_output")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Run pipeline on each PDF in test_pdfs/ and save outputs as baseline in testcases/<basename>/root_output.json."
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=DEFAULT_PDF_DIR,
        help=f"Directory containing test PDFs (default: {DEFAULT_PDF_DIR})",
    )
    parser.add_argument(
        "--testcases-dir",
        type=Path,
        default=DEFAULT_TESTCASES_DIR,
        help=f"Directory to write baseline root_output.json per PDF (default: {DEFAULT_TESTCASES_DIR})",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=RUN_OUTPUT_DIR,
        help=f"Output dir for main.py run (default: {RUN_OUTPUT_DIR})",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent
    pdf_dir = args.pdf_dir if args.pdf_dir.is_absolute() else repo_root / args.pdf_dir
    testcases_dir = args.testcases_dir if args.testcases_dir.is_absolute() else repo_root / args.testcases_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir

    if not pdf_dir.is_dir():
        print(f"Error: PDF directory not found: {pdf_dir}", file=sys.stderr)
        return 1

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}", file=sys.stderr)
        return 1

    for pdf_path in pdfs:
        basename = pdf_path.stem
        print(f"[{basename}] Running pipeline...", flush=True)
        cmd = [
            sys.executable,
            "main.py",
            "--pdf", str(pdf_path.resolve()),
            "-o", str(output_dir.resolve()),
        ]
        result = subprocess.run(cmd, cwd=repo_root)
        if result.returncode != 0:
            print(f"[{basename}] Pipeline failed, skipping.", file=sys.stderr)
            continue
        vocab_path = output_dir / f"{basename}_vocabulary.json"
        if not vocab_path.exists():
            print(f"[{basename}] No vocabulary output at {vocab_path}, skipping.", file=sys.stderr)
            continue
        dest_dir = testcases_dir / basename
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / "root_output.json"
        shutil.copy2(vocab_path, dest_path)
        print(f"[{basename}] Saved baseline to {dest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
