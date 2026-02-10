# Tamil vocabulary extraction (Strands)

PDF or image → OCR → 5-agent graph → vocabulary JSON.

## Requirements

- Python 3.11+
- `pip install -r requirements.txt`

## Environment

- **`GEMINI_API_KEY`** — required. Set in `.env` or export in the environment.

## Run

- **PDF:** `python main.py --pdf path/to/file.pdf`
- **Image:** `python main.py --image path/to/image.png`
- **Output dir:** `-o DIR` (default: `pdf_output`). Writes `<input_basename>_vocabulary.json` (e.g. `test_pdf1_vocabulary.json`) and per-page pipeline logs (`<input_basename>_pageN_pipeline_output.txt`).

Examples:

```bash
python main.py --pdf textbook.pdf
python main.py --image page1.png -o my_output
python main.py --pdf doc.pdf --dpi 300
```

## Docker

No package patch is needed. Install dependencies with `pip install -r requirements.txt`; the app includes an in-project fix for Gemini (empty tools). Pass `GEMINI_API_KEY` as an environment variable when running the container.
