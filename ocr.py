"""
Non-agentic Tamil OCR from images using Gemini.
Extracts Tamil text exactly as it appears (no correction).
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# Use google.generativeai for image-in, text-out (matches your old content_extracter)
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

TAMIL_OCR_PROMPT = """Extract all Tamil text from this image exactly as it appears visually.

Strict Rules:
1. Preserve the EXACT layout, spacing, and line breaks as shown in the image
2. Keep text in Tamil, do NOT translate
3. Keep words with dash(_) as it is in the output .for both single and multiple dash/undercourse (for example: த__காளி, சன்___ல்)
4. Ignore headers, footers, and page numbers
5. Avoid diagrams and tables, only extract readable text
6. Do NOT add any explanations or formatting markers
7. Output ONLY the extracted Tamil text, nothing else

Just output the raw Tamil text exactly as it appears on the page."""


def _ensure_configured() -> None:
    if not GENAI_AVAILABLE:
        raise RuntimeError("Install google-generativeai: pip install google-generativeai")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in .env")
    genai.configure(api_key=api_key)


def tamil_ocr_from_image(image_path: str) -> str:
    """
    Extract Tamil text from an image file exactly as it appears.
    Uses Gemini 2.0 Flash. Non-agentic.

    Args:
        image_path: Path to a PNG/JPEG image (e.g. a textbook page).

    Returns:
        Extracted Tamil text, or an error message string if something fails.
    """
    _ensure_configured()
    image_path = Path(image_path)
    if not image_path.exists():
        return f"Error: File not found: {image_path}"

    try:
        img = Image.open(image_path)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content([TAMIL_OCR_PROMPT, img])
        if response and response.text:
            return response.text.strip()
        return "Error: No text extracted"
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    image_file = sys.argv[1] if len(sys.argv) > 1 else "book_page1.png"
    result = tamil_ocr_from_image(image_file)
    print(result)

    out_path = f"{os.path.splitext(image_file)[0]}_extracted.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"\n✓ Saved to: {out_path}")
