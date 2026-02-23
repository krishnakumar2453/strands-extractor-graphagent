"""
Compare vocabulary before and after validation. For each root that exists in
'before' but not in 'after', check whether all its original forms appear
under some root in 'after'. If any form is missing in 'after', add that root
and its forms to the validated JSON so no words are lost.
"""
import json
import argparse
import sys
from pathlib import Path


def _safe_print(msg: str) -> None:
    """Print to stdout with UTF-8 fallback for Windows."""
    try:
        print(msg)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((msg + "\n").encode("utf-8"))


def all_forms_in(vocab: dict) -> set[str]:
    """Return set of all surface forms present in the vocabulary (any root)."""
    out: set[str] = set()
    for forms in vocab.values():
        if isinstance(forms, list):
            out.update(forms)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover missing roots into validated vocabulary: add roots from 'before' whose forms are not present in 'after'."
    )
    parser.add_argument("--before", required=True, help="Path to vocabulary_before_validation.json")
    parser.add_argument("--after", required=True, help="Path to vocabulary.json (after validation)")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for updated vocabulary (default: same as --after)",
    )
    args = parser.parse_args()
    output_path = Path(args.output) if args.output else Path(args.after)

    with open(args.before, encoding="utf-8") as f:
        before = json.load(f)
    with open(args.after, encoding="utf-8") as f:
        after = json.load(f)

    if not isinstance(before, dict):
        before = {}
    if not isinstance(after, dict):
        after = {}

    forms_after = all_forms_in(after)
    added: list[tuple[str, list[str]]] = []

    for root, forms in before.items():
        if root in after:
            continue
        if not isinstance(forms, list):
            forms = [forms] if forms else []
        missing = [f for f in forms if f not in forms_after]
        if missing:
            after[root] = list(dict.fromkeys(forms))
            added.append((root, forms))

    if added:
        for root, forms in added:
            _safe_print(f"Added root: {root} with forms: {forms}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(after, f, ensure_ascii=False, indent=2)
        _safe_print(f"Wrote updated vocabulary to {output_path} ({len(added)} root(s) added).")
    else:
        _safe_print("No missing roots to add; all forms from 'before' are present in 'after'.")


if __name__ == "__main__":
    main()
