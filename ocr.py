"""
Non-agentic Tamil OCR from images using Gemini.
Extracts Tamil text exactly as it appears (no correction).
Uses the Google Gen AI SDK (google.genai); see https://github.com/googleapis/python-genai
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    from google.genai.types import Part
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    Part = None
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
8. Do not skip or ignore any section and any text in the image.

Just output the raw Tamil text exactly as it appears on the page."""


def _get_client():
    if not GENAI_AVAILABLE:
        raise RuntimeError("Install the Google Gen AI SDK: pip install google-genai")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in .env")
    return genai.Client(api_key=api_key)


def tamil_ocr_from_image(image_path: str) -> str:
    """
    Extract Tamil text from an image file exactly as it appears.
    Uses Gemini 2.0 Flash via the Google Gen AI SDK. Non-agentic.

    Args:
        image_path: Path to a PNG/JPEG image (e.g. a textbook page).

    Returns:
        Extracted Tamil text, or an error message string if something fails.
    """
    _get_client()  # raise if not configured
    image_path = Path(image_path)
    if not image_path.exists():
        return f"Error: File not found: {image_path}"

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        suffix = image_path.suffix.lower()
        mime_type = "image/png" if suffix == ".png" else "image/jpeg"
        client = _get_client()
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                TAMIL_OCR_PROMPT,
                Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
        )
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
