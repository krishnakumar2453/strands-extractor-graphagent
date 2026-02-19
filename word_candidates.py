"""
OCR to word-candidates pipeline (deterministic).
Reads an OCR text file and returns a filtered list of Tamil word candidates.
"""

import re
import unicodedata
from pathlib import Path

# Tamil Unicode block
TAMIL_START, TAMIL_END = 0x0B80, 0x0BFF


def _tamil_grapheme_count(s: str) -> int:
    """Count Tamil letters (base characters) in s. One grapheme = one Lo in Tamil block."""
    return sum(
        1
        for c in s
        if TAMIL_START <= ord(c) <= TAMIL_END and unicodedata.category(c) == "Lo"
    )


def _is_tamil_only(s: str) -> bool:
    """True if s is non-empty and every code point is in Tamil block (letters/marks/numbers)."""
    if not s:
        return False
    return all(TAMIL_START <= ord(c) <= TAMIL_END for c in s)


def _strip_to_tamil(s: str) -> str:
    """Return the substring of s that contains only Tamil script (for counting graphemes)."""
    return "".join(c for c in s if TAMIL_START <= ord(c) <= TAMIL_END)


def _is_single_char_token(token: str) -> bool:
    """True if token is 'single character' for rules 1.1/1.2/1.3 (Option A: strip symbols first)."""
    tamil_part = _strip_to_tamil(token)
    return _tamil_grapheme_count(tamil_part) == 1


def _is_valid_word(token: str) -> bool:
    """Valid word = has at least 2 Tamil graphemes (and no underscore at this stage)."""
    return _tamil_grapheme_count(_strip_to_tamil(token)) >= 2


def _is_single_char_with_special_only(token: str) -> bool:
    """True if token is only special chars and single Tamil grapheme(s) (e.g. (இ), [மா ,சி])."""
    tamil_runs = re.findall(r"[\u0B80-\u0BFF]+", token)
    if not tamil_runs:
        return False
    for run in tamil_runs:
        if _tamil_grapheme_count(run) >= 2:
            return False
    return True


def _tokenize_lines(content: str) -> list[list[str]]:
    """Split content into lines, then each line into tokens by whitespace. Preserves structure."""
    lines = content.split("\n")
    return [re.split(r"\s+", line) for line in lines]


def _flatten_tokens_with_line_info(line_tokens: list[list[str]]) -> list[tuple[str, int]]:
    """Flatten to (token, line_index) list; empty tokens from split dropped."""
    flat = []
    for line_idx, tokens in enumerate(line_tokens):
        for t in tokens:
            if t:
                flat.append((t, line_idx))
    return flat


def _remove_tokens_with_underscore(line_tokens: list[list[str]]) -> list[list[str]]:
    """Step 2: Remove any token that contains underscore."""
    return [[t for t in line if "_" not in t] for line in line_tokens]


def _step_1_1_single_char_only_lines(line_tokens: list[list[str]]) -> list[list[str]]:
    """Step 1.1: On each line, if the only token(s) are single-char, remove them (empty line)."""
    out = []
    for line in line_tokens:
        valid_or_multi = [t for t in line if not _is_single_char_token(t)]
        if valid_or_multi:
            out.append(line)  # keep line as-is; we only drop single-char-only lines
        else:
            out.append([])  # all single-char or empty -> empty line
    return out


def _step_1_2_remove_series(flat: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Step 1.2: Remove runs of 2+ consecutive single-char tokens."""
    if not flat:
        return flat
    result = []
    i = 0
    while i < len(flat):
        token, line_idx = flat[i]
        if not _is_single_char_token(token):
            result.append(flat[i])
            i += 1
            continue
        # Find run of single-char tokens
        j = i
        while j < len(flat) and _is_single_char_token(flat[j][0]):
            j += 1
        # Run length = j - i. If >= 2, drop entire run; else keep single one (handled below: we don't keep)
        if j - i >= 2:
            i = j
            continue
        # Single isolated single-char token: keep for now (1.3 will decide)
        result.append(flat[i])
        i += 1
    return result


def _step_1_3_remove_single_unless_adjacent_valid(flat: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Step 1.3 (Option A): Remove single-char tokens unless prev or next is a valid word."""
    if not flat:
        return flat
    result = []
    for i, (token, line_idx) in enumerate(flat):
        if not _is_single_char_token(token):
            result.append(flat[i])
            continue
        prev_ok = i > 0 and _is_valid_word(flat[i - 1][0])
        next_ok = i + 1 < len(flat) and _is_valid_word(flat[i + 1][0])
        if prev_ok or next_ok:
            result.append(flat[i])
    return result


def _replace_special_with_space(s: str) -> str:
    """Step 3: Replace punctuation/symbols (keyboard + Tamil punctuation) with space."""
    # Keep Tamil letters, digits, whitespace; replace rest with space
    return re.sub(r"[^\s\u0B80-\u0BFF0-9]", " ", s)


def _replace_numbers_with_space(s: str) -> str:
    """Step 4: Replace digits with space."""
    return re.sub(r"[0-9]", " ", s)


def _split_whitespace(s: str) -> list[str]:
    """Split on any run of whitespace (including newlines)."""
    return re.split(r"\s+", s)


def _remove_single_char_series_from_tokens(tokens: list[str]) -> list[str]:
    """Remove runs of 2+ consecutive single-char tokens from a list of tokens."""
    if not tokens:
        return tokens
    result = []
    i = 0
    while i < len(tokens):
        if not _is_single_char_token(tokens[i]):
            result.append(tokens[i])
            i += 1
            continue
        j = i
        while j < len(tokens) and _is_single_char_token(tokens[j]):
            j += 1
        if j - i >= 2:
            i = j
            continue
        result.append(tokens[i])
        i += 1
    return result


def _step_5_tamil_only(tokens: list[str]) -> list[str]:
    """Step 5: Keep only tokens that are entirely Tamil (and non-empty)."""
    return [t for t in tokens if t and _is_tamil_only(t)]


def _run_pipeline(
    content: str,
    deduplicate: bool,
) -> tuple[str, list[str]]:
    """Run full pipeline on OCR content. Returns (pre_split_text, candidates)."""
    line_tokens = _tokenize_lines(content)
    line_tokens = _remove_tokens_with_underscore(line_tokens)
    line_tokens = _step_1_1_single_char_only_lines(line_tokens)
    flat = _flatten_tokens_with_line_info(line_tokens)
    flat = _step_1_2_remove_series(flat)
    # Before 1.3: replace single-char + special tokens with space (e.g. (இ), [மா ,சி])
    flat = [
        (" ", idx) if _is_single_char_with_special_only(t) else (t, idx)
        for t, idx in flat
    ]
    flat = _step_1_3_remove_single_unless_adjacent_valid(flat)

    text = " ".join(t for t, _ in flat)
    text = _replace_special_with_space(text)
    text = _replace_numbers_with_space(text)
    pre_split_text = text

    # Before final split: remove any single-char series in the cleaned OCR string
    tokens_pre = _split_whitespace(pre_split_text)
    tokens_pre = [t for t in tokens_pre if t]
    tokens_pre = _remove_single_char_series_from_tokens(tokens_pre)
    pre_split_text = " ".join(tokens_pre)

    tokens = _split_whitespace(pre_split_text)
    candidates = _step_5_tamil_only(tokens)

    if deduplicate:
        seen: set[str] = set()
        unique: list[str] = []
        for w in candidates:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        candidates = unique

    return (pre_split_text, candidates)


def ocr_path_to_word_candidates(
    ocr_txt_path: str | Path,
    encoding: str = "utf-8",
    deduplicate: bool = True,
) -> list[str]:
    """
    Read OCR file, apply filter pipeline, return ordered list of Tamil word candidates.

    Algorithm (order):
      1. Read file, split into lines, tokenize each line by whitespace.
      2. Remove tokens containing underscore.
      1.1 Remove single-char tokens when they are the only content on that line.
      1.2 Remove runs of 2+ consecutive single-char tokens.
      1.3 Remove single-char tokens unless adjacent to a valid word (Option A).
      3. Replace punctuation/symbols with space and re-tokenize.
      4. Replace numbers with space and re-tokenize.
      5. Keep only tokens that are entirely Tamil script.

    Input: path to OCR text file (e.g. pdf_output/test_pdf1_full_ocr.txt).
    Output: list of word strings in order. If deduplicate is True (default), first occurrence is kept.
    """
    content = Path(ocr_txt_path).read_text(encoding=encoding)
    _, candidates = _run_pipeline(content, deduplicate)
    return candidates


def ocr_path_to_word_candidates_with_insight(
    ocr_txt_path: str | Path,
    encoding: str = "utf-8",
    deduplicate: bool = True,
) -> tuple[str, list[str]]:
    """
    Same pipeline as ocr_path_to_word_candidates, but also return the OCR string
    *before* the final split (after steps 2, 1.1–1.3, 3, 4: punctuation and numbers
    replaced by space, only kept tokens). Splitting this string on whitespace and
    keeping Tamil-only tokens yields the candidate list.

    Returns:
        (pre_split_text, candidates)
        - pre_split_text: string containing only Tamil letters and spaces.
        - candidates: list of word strings (same as ocr_path_to_word_candidates).
    """
    content = Path(ocr_txt_path).read_text(encoding=encoding)
    return _run_pipeline(content, deduplicate)


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python ocr_word_candidates.py <ocr.txt path> [candidates.json] [pre_split.txt]\n"
            "  pre_split.txt: optional; write OCR string before final split (Tamil + spaces only)."
        )
        sys.exit(1)
    path = sys.argv[1]
    out_json = sys.argv[2] if len(sys.argv) > 2 else None
    out_pre_split = sys.argv[3] if len(sys.argv) > 3 else None

    if out_pre_split is not None:
        pre_split_text, candidates = ocr_path_to_word_candidates_with_insight(path)
        Path(out_pre_split).write_text(pre_split_text, encoding="utf-8")
    else:
        candidates = ocr_path_to_word_candidates(path)

    json_str = json.dumps(candidates, ensure_ascii=False, indent=0)
    if out_json:
        Path(out_json).write_text(json_str, encoding="utf-8")
    else:
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
        print(json_str)
