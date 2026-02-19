"""
Non-agentic PDF → images → Tamil OCR.
Converts each page to an image, runs OCR, returns combined text.
"""
import os
import sys
from pathlib import Path

import fitz  # PyMuPDF

from ocr import tamil_ocr_from_image

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def pdf_to_ocr_text(
    pdf_path: str,
    output_dir: str | None = "pdf_output",
    dpi: int = 300,
    save_page_images: bool = False,
) -> str:
    """
    Convert a PDF to images page by page, run Tamil OCR on each, return combined OCR text.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory for temporary page images (and optional saved images). Created if missing.
        dpi: Resolution for rendering pages (default 300).
        save_page_images: If True, keep page images on disk; if False, delete after OCR.

    Returns:
        Combined OCR text with "--- Page N ---" markers between pages.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    os.makedirs(output_dir, exist_ok=True)
    pdf = fitz.open(pdf_path)
    pages = len(pdf)
    all_ocr_text = []

    for page_num in range(pages):
        print(f"Processing page {page_num + 1}/{pages} ...", flush=True)
        page = pdf[page_num]
        pix = page.get_pixmap(dpi=dpi)
        temp_image = os.path.join(output_dir, f"page_{page_num + 1}.png")
        pix.save(temp_image)

        page_text = tamil_ocr_from_image(temp_image)
        if page_text.startswith("Error:"):
            print(f"  Warning: {page_text}", flush=True)
        all_ocr_text.append(f"--- Page {page_num} ---\n{page_text}")

        if not save_page_images:
            try:
                os.remove(temp_image)
            except OSError:
                pass

    pdf.close()

    return "\n\n".join(all_ocr_text)


def get_ocr_per_page(
    pdf_path: str,
    output_dir: str | None = "pdf_output",
    dpi: int = 300,
) -> list[str]:
    """
    Convert a PDF to images page by page, run Tamil OCR on each, return one OCR string per page.

    Returns:
        List of OCR text strings; index i is page i (0-based in list, 1-based for display).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    os.makedirs(output_dir, exist_ok=True)
    pdf = fitz.open(pdf_path)
    pages = len(pdf)
    page_texts = []

    for page_num in range(pages):
        print(f"Processing page {page_num + 1}/{pages} ...", flush=True)
        page = pdf[page_num]
        pix = page.get_pixmap(dpi=dpi)
        temp_image = os.path.join(output_dir, f"page_{page_num + 1}.png")
        pix.save(temp_image)

        page_text = tamil_ocr_from_image(temp_image)
        if page_text.startswith("Error:"):
            print(f"  Warning: {page_text}", flush=True)
        page_texts.append(page_text.strip())

        try:
            os.remove(temp_image)
        except OSError:
            pass

    pdf.close()
    return page_texts


def process_pdf(
    pdf_path: str,
    output_dir: str = "pdf_output",
    dpi: int = 300,
    save_ocr_txt: bool = True,
) -> str:
    """
    Process a PDF: render pages, run Tamil OCR, optionally save combined OCR to a text file.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory for output files (and temp images). Default "pdf_output".
        dpi: Resolution for page images. Default 300.
        save_ocr_txt: If True, save combined OCR to {pdf_name}_full_ocr.txt in output_dir.

    Returns:
        Combined OCR text (with "--- Page N ---" markers).
    """
    combined_ocr = pdf_to_ocr_text(pdf_path, output_dir=output_dir, dpi=dpi, save_page_images=False)

    if save_ocr_txt:
        pdf_name = Path(pdf_path).stem
        ocr_file = os.path.join(output_dir, f"{pdf_name}_full_ocr.txt")
        with open(ocr_file, "w", encoding="utf-8") as f:
            f.write(combined_ocr)
        print(f"✓ OCR saved to: {ocr_file}")

    return combined_ocr


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract Tamil OCR text from a PDF (non-agentic).")
    parser.add_argument("pdf_file", help="Path to the PDF file")
    parser.add_argument("-o", "--output-dir", default="pdf_output", help="Output directory (default: pdf_output)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for page images (default: 300)")
    args = parser.parse_args()

    process_pdf(args.pdf_file, output_dir=args.output_dir, dpi=args.dpi)
