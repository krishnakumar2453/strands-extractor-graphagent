"""
OCR validation: compare image with LLM-returned OCR text.
If any word, full stop (.), comma (,), or underscore (_) is missing, return corrected OCR.
Otherwise return exactly "NOTHING MISS".
Uses Gemini with a strict validation prompt.
"""
from pathlib import Path

from PIL import Image

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from ocr import _ensure_configured


VALIDATE_OCR_PROMPT = """You are a strict Tamil OCR validator. You will be given:
1. An image containing Tamil text (e.g. textbook page).
2. The CURRENT OCR TEXT that was extracted from this image (possibly with omissions).

Your ONLY job is to compare the image with the OCR text and validate:

— Are ANY of the following MISSING in the OCR text but clearly visible in the image?
  • Any whole word (Tamil word)
  • Full stop (.)
  • Comma (,)

If ANYTHING is missing:
  • Output the COMPLETE corrected OCR text: the full text with every missing word, full stop, comma ADDED in the correct position.
  • Preserve exact layout, line breaks, and spacing as in the image.
  • Do NOT add any prefix, explanation, or commentary. Do NOT say "Here is the corrected text" or similar. Output ONLY the raw corrected Tamil text.

If NOTHING is missing (the OCR text matches the image for all words, full stops, commas, and underscores):
  • Output exactly and only these two words, with a single space, no other characters: NOTHING MISS

STRICT RULES:
1. Only ADD what is clearly visible in the image. Do not invent or guess.
2. Do not REMOVE, REORDER, or REWRITE existing correct text. Only insert missing elements.
3. Do not translate. Keep all text in Tamil as in the image.
4. Your response must be either (a) exactly "NOTHING MISS" or (b) the complete corrected OCR text and nothing else — no markdown, no quotes, no extra lines before or after.
"""


def validate_ocr_from_image(image_path: str | Path, ocr_text: str) -> str:
    """
    Validate OCR text against the image: add any missing words, full stops, commas, or underscores.

    Args:
        image_path: Path to the image file (same image that was used for OCR).
        ocr_text: The OCR text returned by the LLM (e.g. contents of _full_ocr.txt).

    Returns:
        Either the literal string "NOTHING MISS" if nothing is missing, or the complete
        corrected OCR text with all missing elements added.
        On error, returns a string starting with "Error:".
    """
    _ensure_configured()
    if not GENAI_AVAILABLE:
        return "Error: Install google-generativeai: pip install google-generativeai"

    image_path = Path(image_path)
    if not image_path.exists():
        return f"Error: File not found: {image_path}"

    ocr_text = (ocr_text or "").strip()
    prompt_with_ocr = (
        VALIDATE_OCR_PROMPT
        + "\n\n--- CURRENT OCR TEXT TO VALIDATE ---\n"
        + ocr_text
        + "\n--- END OCR TEXT ---\n\n"
        + "Compare the image above with the OCR text. If anything is missing, output the full corrected OCR. Otherwise output exactly: NOTHING MISS"
    )

    try:
        img = Image.open(image_path)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content([img, prompt_with_ocr])
        if not response or not response.text:
            return "Error: No response from validator"

        out = response.text.strip()
        if out.upper() == "NOTHING MISS":
            return "NOTHING MISS"
        return out
    except Exception as e:
        return f"Error: {str(e)}"
