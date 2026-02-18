# Tamil word candidate extraction from OCR text (no LLM).
# Replaces word_candidate_agent + word_candidate_validator_agent with deterministic filtering.
import re
import json
import sys
import argparse
from pathlib import Path

# Tamil Unicode block (letters, marks, digits U+0B80–U+0BFF)
TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
# Keep only Tamil script in output (remove all other symbols/letters/digits)
TAMIL_ONLY_RE = re.compile(r"[^\u0B80-\u0BFF]")

# Strip these from start/end of tokens to get "core" word.
# Includes: space, . ? ! , ' " ( ) _ - [ ] : ; and curly quotes U+2018 ‘ U+2019 ’ U+201C " U+201D "
# We split on whitespace only; full stop stays inside tokens so we can detect initials (e.g. வே.சாமிநாதர்).
EDGE_SYMBOLS_RE = re.compile(
    r"^[\s.,?!'\"()_\-\[\]:;\u2018\u2019\u201C\u201D]+|[\s.,?!'\"()_\-\[\]:;\u2018\u2019\u201C\u201D]+$"
)


def _core(token: str) -> str:
    """Strip leading/trailing symbols/punctuation to get the word core."""
    return EDGE_SYMBOLS_RE.sub("", token)


def _is_single_char_token(token: str, max_len: int = 2) -> bool:
    """Treat 1–2 codepoint tokens as one Tamil letter (one grapheme)."""
    return 1 <= len(token) <= max_len


def _remove_single_char_runs(
    tokens: list[str], cores: list[str], max_single_len: int = 2
) -> list[str]:
    """Remove only runs of 2+ consecutive single-char tokens (a series).
    A single-char token with no single-char next or previous (standalone) is kept, e.g. பூ in இந்த பூ அழகானது.
    """
    if not tokens:
        return []
    single = {
        i
        for i, c in enumerate(cores)
        if _is_single_char_token(c, max_single_len)
    }
    to_drop = set()
    i = 0
    while i < len(tokens):
        if i not in single:
            i += 1
            continue
        j = i
        while j < len(tokens) and j in single:
            j += 1
        if j - i >= 2:
            to_drop.update(range(i, j))
        i = j
    return [t for k, t in enumerate(tokens) if k not in to_drop]


def _single_char_run_drop_indices(cores: list[str], max_single_len: int = 2) -> set[int]:
    """Return set of indices that are part of a run of 2+ single-char tokens (to drop)."""
    if not cores:
        return set()
    single = {i for i, c in enumerate(cores) if _is_single_char_token(c, max_single_len)}
    to_drop = set()
    i = 0
    while i < len(cores):
        if i not in single:
            i += 1
            continue
        j = i
        while j < len(cores) and j in single:
            j += 1
        if j - i >= 2:
            to_drop.update(range(i, j))
        i = j
    return to_drop


def _is_english_only(token: str) -> bool:
    """True if token is entirely or majority ASCII letters."""
    if not token:
        return True
    letters = sum(1 for c in token if c.isalpha() and ord(c) < 128)
    return letters >= (len(token) + 1) // 2


def _is_symbol_or_number(token: str) -> bool:
    """Drop pure numbers, decimals, and symbol-only tokens."""
    s = token.strip()
    if not s:
        return True
    if s.isdigit():
        return True
    if re.match(r"^[\d.,]+$", s):
        return True
    if all(not c.isalnum() and ord(c) < 0x80 for c in s):
        return True
    return False


def _has_tamil(s: str) -> bool:
    return bool(TAMIL_RE.search(s))


def _tamil_only(s: str) -> str:
    """Remove all non-Tamil characters. Return only Tamil script (safer: no symbols in output)."""
    return TAMIL_ONLY_RE.sub("", s)


def _expand_token_drop_initials(core: str, max_single_len: int = 2) -> list[str]:
    """If core contains '.', split on '.' and drop leading single-char parts that are followed by a long (3+ char) part (name initials). Return list of parts to keep."""
    if "." not in core:
        return [core] if core else []
    parts = [p.strip() for p in core.split(".") if p.strip()]
    if not parts:
        return []
    # Find first long part (3+ chars)
    i = 0
    while i < len(parts) and _is_single_char_token(parts[i], max_single_len):
        i += 1
    # If we have a long part at i, drop parts[0:i] (initials). If all parts are short, keep all.
    if i < len(parts) and len(parts[i]) >= 3:
        return parts[i:]
    return parts


def get_word_candidates_from_ocr_file(
    ocr_txt_path: str | Path,
    *,
    remove_single_char_runs: bool = True,
    max_single_char_len: int = 2,
    drop_english: bool = True,
    require_tamil: bool = True,
    dedupe: bool = False,
) -> list[str]:
    """
    Read OCR text from file, split on whitespace, filter, and return word candidate list.

    - Removes tokens containing '__'.
    - Removes pure numbers/symbols.
    - Removes English (or non-Tamil) words when requested.
    - Removes entire runs of single-character tokens (e.g. OCR line-by-line fragments).
    """
    path = Path(ocr_txt_path)
    text = path.read_text(encoding="utf-8")
    return get_word_candidates_from_ocr_text(
        text,
        remove_single_char_runs=remove_single_char_runs,
        max_single_char_len=max_single_char_len,
        drop_english=drop_english,
        require_tamil=require_tamil,
        dedupe=dedupe,
    )


def get_word_candidates_from_ocr_text(
    ocr_text: str,
    *,
    remove_single_char_runs: bool = True,
    max_single_char_len: int = 2,
    drop_english: bool = True,
    require_tamil: bool = True,
    dedupe: bool = False,
) -> list[str]:
    """
    Same as get_word_candidates_from_ocr_file but accepts the OCR string directly.
    Pipeline order:

    1. Split on whitespace only (space, \\n, tabs). Full stop and ? stay inside tokens.
    2. Strip edge symbols (., ?, etc.) to get core; for tokens containing '.', drop name initials
       (leading single-char parts before a long 3+ char part), e.g. வே.சாமிநாதர் → keep only சாமிநாதர்.
    3. Single-char series: remove only runs of 2+ consecutive single-char tokens; keep standalone (e.g. பூ).
    4. Drop empty; drop single-char with symbols (e.g. (இ), ஆ!); drop __; drop numbers/symbols; drop English; require Tamil.
    5. Output Tamil-only (no symbols in candidates).
    """
    # 1. Split on whitespace only (full stop and ? stay inside tokens)
    raw_tokens = [t for t in re.split(r"\s+", ocr_text) if t]
    # 2. Expand: strip edges, then for tokens with '.', drop initials and keep remaining parts
    flat: list[tuple[str, str]] = []
    for t in raw_tokens:
        core = _core(t)
        if not core:
            continue
        for part in _expand_token_drop_initials(core, max_single_char_len):
            flat.append((part, t))
    if not flat:
        return []
    parts = [p for p, _ in flat]
    # 3. Single-char run removal (only runs of 2+)
    if remove_single_char_runs:
        to_drop = _single_char_run_drop_indices(parts, max_single_char_len)
        flat = [(p, o) for i, (p, o) in enumerate(flat) if i not in to_drop]
    # 4 & 5. Apply filters in order, then Tamil-only output
    out = []
    for part, orig_token in flat:
        if not part:
            continue
        if _is_single_char_token(part, max_single_char_len) and TAMIL_ONLY_RE.search(orig_token):
            continue
        if "__" in part:
            continue
        if _is_symbol_or_number(part):
            continue
        if drop_english and _is_english_only(part):
            continue
        if require_tamil and not _has_tamil(part):
            continue
        candidate = _tamil_only(part)
        if not candidate:
            continue
        out.append(candidate)
    if dedupe:
        out = list(dict.fromkeys(out))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Tamil word candidates from an OCR text file (no LLM)."
    )
    parser.add_argument(
        "ocr_txt",
        metavar="OCR_TXT_PATH",
        help="Path to OCR text file (e.g. pdf_output/test_pdf1_full_ocr.txt)",
    )
    parser.add_argument(
        "-o", "--output",
        metavar="PATH",
        help="Write JSON list to file (default: print to stdout)",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep duplicate words (default: dedupe)",
    )
    parser.add_argument(
        "--no-single-char-removal",
        action="store_true",
        help="Do not remove runs of single-character tokens",
    )
    args = parser.parse_args()

    candidates = get_word_candidates_from_ocr_file(
        args.ocr_txt,
        remove_single_char_runs=not args.no_single_char_removal,
        dedupe=not args.no_dedupe,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(candidates, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {len(candidates)} word candidates to {out_path}")
    else:
        # Use UTF-8 for stdout so Tamil prints on Windows
        out = json.dumps(candidates, ensure_ascii=False, indent=2)
        if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8")
        print(out)


if __name__ == "__main__":
    main()
