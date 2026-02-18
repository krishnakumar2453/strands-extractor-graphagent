"""
Baseline comparison: run pipeline on test_pdfs/*.pdf and compare outputs to
testcases/<basename>/root_output.json. Report only missed roots, extra roots,
and differing form lists.
"""
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_PDF_DIR = Path("test_pdfs")
DEFAULT_TESTCASES_DIR = Path("testcases")
COMPARE_OUTPUT_DIR = Path("compare_output")


def discover_pdfs(pdf_dir: Path) -> list[Path]:
    """Return sorted list of PDF paths in pdf_dir."""
    if not pdf_dir.is_dir():
        return []
    return sorted(pdf_dir.glob("*.pdf"))


def run_pipeline(pdf_path: Path, output_dir: Path) -> bool:
    """Run main.py on pdf_path, writing to output_dir. Return True on success."""
    cmd = [
        sys.executable,
        "main.py",
        "--pdf", str(pdf_path.resolve()),
        "-o", str(output_dir.resolve()),
    ]
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent)
    return result.returncode == 0


def load_vocabulary(path: Path) -> dict[str, list[str]] | None:
    """Load root -> [forms] JSON. Return None if file missing or invalid."""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return {k: list(v) if isinstance(v, list) else [] for k, v in data.items()}
    except (json.JSONDecodeError, OSError):
        return None


def compare_vocabularies(
    baseline: dict[str, list[str]],
    new: dict[str, list[str]],
) -> tuple[list[str], list[str], list[tuple[str, list[str], list[str]]]]:
    """
    Compare baseline vs new. Return (missed_roots, extra_roots, different).
    different is list of (root, missed_forms, extra_forms).
    """
    baseline_roots = set(baseline)
    new_roots = set(new)
    missed_roots = sorted(baseline_roots - new_roots)
    extra_roots = sorted(new_roots - baseline_roots)
    different: list[tuple[str, list[str], list[str]]] = []
    for root in sorted(baseline_roots & new_roots):
        b_forms = set(baseline[root])
        n_forms = set(new[root])
        if b_forms != n_forms:
            missed_forms = sorted(b_forms - n_forms)
            extra_forms = sorted(n_forms - b_forms)
            different.append((root, missed_forms, extra_forms))
    return missed_roots, extra_roots, different


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Run pipeline on test PDFs and compare to baseline (testcases/<basename>/root_output.json)."
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
        help=f"Directory containing baseline root_output.json per PDF (default: {DEFAULT_TESTCASES_DIR})",
    )
    parser.add_argument(
        "--pdf",
        metavar="BASENAME",
        help="Compare only this PDF (basename, e.g. test_pdf1). File must be in --pdf-dir. Omit to compare all PDFs.",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent
    pdf_dir = (args.pdf_dir if args.pdf_dir.is_absolute() else repo_root / args.pdf_dir)
    testcases_dir = (args.testcases_dir if args.testcases_dir.is_absolute() else repo_root / args.testcases_dir)
    output_dir = repo_root / COMPARE_OUTPUT_DIR

    pdfs = discover_pdfs(pdf_dir)
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}", file=sys.stderr)
        return 1
    if args.pdf:
        pdfs = [p for p in pdfs if p.stem == args.pdf]
        if not pdfs:
            print(f"No PDF found with basename '{args.pdf}' in {pdf_dir}. Use the filename without .pdf (e.g. test_pdf1).", file=sys.stderr)
            return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    any_fail = False
    any_skip = False

    for pdf_path in pdfs:
        basename = pdf_path.stem
        baseline_path = testcases_dir / basename / "root_output.json"
        baseline = load_vocabulary(baseline_path)
        if baseline is None:
            print(f"[{basename}] SKIP: no baseline at {baseline_path}", file=sys.stderr)
            any_skip = True
            continue

        print(f"[{basename}] Running pipeline...", flush=True)
        if not run_pipeline(pdf_path, output_dir):
            print(f"[{basename}] FAIL: pipeline exited non-zero", file=sys.stderr)
            any_fail = True
            continue

        new_path = output_dir / f"{basename}_vocabulary.json"
        new = load_vocabulary(new_path)
        if new is None:
            print(f"[{basename}] FAIL: no output at {new_path}", file=sys.stderr)
            any_fail = True
            continue

        missed_roots, extra_roots, different = compare_vocabularies(baseline, new)
        if not missed_roots and not extra_roots and not different:
            print(f"[{basename}] OK")
            continue

        # Build diff report (print and save to compare_output)
        report_lines = [
            f"[{basename}] DIFF:",
        ]
        if missed_roots:
            report_lines.append(f"  missed roots: {missed_roots}")
        if extra_roots:
            report_lines.append(f"  extra roots: {extra_roots}")
        if different:
            for root, missed_forms, extra_forms in different:
                report_lines.append(f"  different '{root}': missed_forms {missed_forms}, extra_forms {extra_forms}")
        report_text = "\n".join(report_lines)
        for line in report_lines:
            print(line)
        diff_path = output_dir / f"{basename}_diff.txt"
        try:
            diff_path.write_text(report_text, encoding="utf-8")
            print(f"  (saved to {diff_path})")
        except OSError as e:
            print(f"  (failed to save diff: {e})", file=sys.stderr)
        if missed_roots or different:
            any_fail = True

    if any_skip:
        print("Some PDFs skipped (no baseline). Add baselines with save_baseline.py.", file=sys.stderr)
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
