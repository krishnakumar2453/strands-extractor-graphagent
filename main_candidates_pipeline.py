# Tamil vocabulary extraction pipeline — word_candidates.py + 3-agent graph
# Word candidates from word_candidates.py. Then: noise_filter → root_normalizer (3 tools) → variant_grouping.
# Root normalizer: tool1 normalize_to_root → tool2 validate_root_forms → if needs_fix, tool3 fix_roots_in_list.
from strands.multiagent import GraphBuilder
from strands import Agent, tool
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv
from google import genai
from strands.models.gemini import GeminiModel

load_dotenv()

# --- LLM helper for tools ---
def _call_llm_json(system: str, user_content: str) -> str:
    """Call Gemini with system + user content; return raw response text."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "{}"
    client = genai.Client(api_key=api_key)
    config = genai.types.GenerateContentConfig(
        system_instruction=system,
        temperature=0,
        response_mime_type="application/json",
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=user_content,
        config=config,
    )
    return (response.text or "").strip() or "{}"


# --- Tool 1: normalize_to_root — remove suffixes and normalize to Tamil dictionary root form ---
NORMALIZE_SYSTEM = """You convert Tamil words to their dictionary base/root form. Output one {"root", "form"} pair per input word.

RULE — WHEN TO STRIP A SUFFIX:
Strip an ending ONLY when BOTH are true:
  (a) The ending is a known grammatical suffix (see list below), AND
  (b) After removing it, the REMAINING part is itself a valid Tamil dictionary word (base form).

DO NOT STRIP when the remainder would not be a real word. Example: மகள் (daughter) → root must be மகள். The ending கள் here is part of the word; if you strip it you get ம, which is not a word. So NEVER output ம for மகள்.

Grammatical suffixes (strip only when the stem is a real word): கள் (plural), ஐ (accusative), இல் (locative), க்கு (dative), ஆல் (instrumental), உடன் (with), என்று (that/saying), ஆக (as), ஆம் (affirmative), ஒடு/ஓடு (with), அது (that), வாறு (manner); word-final sandhi ச், ப், த் when they are suffix markers (e.g. before இல், ஆல்). Also: ற்றுள், களுள் when clearly locative/plural suffix.

EXAMPLES:
- அரசர்கள் → அரசன் (கள் is plural; அரசன் is dictionary word).
- வகுப்பில் → வகுப்பு (இல் is locative).
- மகள் → மகள் (do NOT strip; ம is not a word).
- நிலங்கள் → நிலம் (கள் is plural; நிலம் is word).
- மொழியொடு → மொழி (ஒடு is "with").

CRITICAL — NOT A SINGLE WORD MUST BE MISSED: Words are very important. You must return exactly one pair per input word. Total number of "form" values in your output MUST equal the number of input words. Every input word must appear exactly once as "form". Do not drop, merge, or add any form. Count input words and count output "form" entries; they must be equal.

OUTPUT (this exact JSON only):
{"normalized": [{"root": "root_word", "form": "original_form"}, ...]}
Return ONLY valid JSON, no other text."""

@tool
def normalize_to_root(words: list[str]) -> dict[str, Any]:
    """Remove grammatical suffixes and normalize all words to Tamil dictionary root form.

    Args:
        words: List of Tamil words (e.g. filtered_candidates from noise_filter).

    Returns:
        Tool result with "normalized": [{"root": "...", "form": "..."}, ...].
    """
    if not words:
        return {"status": "success", "content": [{"text": json.dumps({"normalized": []}, ensure_ascii=False)}]}
    user_content = "Normalize these Tamil words to root form. Return JSON with key \"normalized\" and a list of {\"root\", \"form\"} pairs.\nInput words:\n" + json.dumps(
        words, ensure_ascii=False
    )
    out = _call_llm_json(NORMALIZE_SYSTEM, user_content)
    try:
        data = json.loads(out)
        if "normalized" not in data:
            data = {"normalized": data} if isinstance(data, list) else {"normalized": []}
    except json.JSONDecodeError:
        data = {"normalized": [], "raw": out[:500]}
    return {"status": "success", "content": [{"text": json.dumps(data, ensure_ascii=False)}]}


# --- Tool 2: validate_root_forms — check if all roots are in correct dictionary form ---
VALIDATE_SYSTEM = """You validate a list of Tamil root/form pairs. For each pair, check if "root" is in correct dictionary root form (no grammatical suffix left on the root).

RULE — WHEN IS A ROOT CORRECT:
The root is correct if it is a dictionary base form: either it has no grammatical suffix, or the ending is NOT a suffix but part of the word itself. Example: மகள் as root is CORRECT (கள் is part of the word "daughter"; ம is not a word). Do NOT flag மகள். Example: அரசர்கள் as root is WRONG (கள் here is plural suffix; correct root is அரசன்). Flag it.

Grammatical suffixes that must NOT remain on the root (when the stem without them is a real word): கள், ஐ, இல், க்கு, ஆல், உடன், என்று, ஆக, ஆம், ஒடு, ஓடு, அது, வாறு, ற்றுள், களுள்; word-final sandhi ச், ப், த் when suffix. Do NOT list roots like மகள் where the ending is part of the word.

If ALL roots are in correct form, return exactly:
{"status": "all_correct", "message": "ALL ARE IN CORRECT FORM"}
If ANY root still has a grammatical suffix (and stripping it would leave a valid word), return exactly:
{"status": "needs_fix", "roots_to_fix": ["root1", "root2", ...]}
List only the root strings that need correction. Return ONLY valid JSON, no other text."""


@tool
def validate_root_forms(normalized: list[dict]) -> dict[str, Any]:
    """Validate whether all roots in the normalized list are in correct dictionary root form (no grammatical suffixes).

    Args:
        normalized: List of {"root": "...", "form": "..."} from normalize_to_root.

    Returns:
        If all correct: {"status": "all_correct", "message": "ALL ARE IN CORRECT FORM"}.
        If some need fixing: {"status": "needs_fix", "roots_to_fix": ["root_a", "root_b", ...]}.
    """
    if not normalized:
        return {"status": "success", "content": [{"text": json.dumps({"status": "all_correct", "message": "ALL ARE IN CORRECT FORM"}, ensure_ascii=False)}]}
    user_content = "Validate these pairs. Is each \"root\" in dictionary base form (no grammatical suffix)?\n" + json.dumps(
        {"normalized": normalized}, ensure_ascii=False
    )
    out = _call_llm_json(VALIDATE_SYSTEM, user_content)
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        data = {"status": "needs_fix", "roots_to_fix": []}
    return {"status": "success", "content": [{"text": json.dumps(data, ensure_ascii=False)}]}


# --- Tool 3: fix_roots_in_list — correct specified roots in the normalized list ---
FIX_ROOTS_SYSTEM = """You correct only the roots listed in roots_to_fix. For each pair whose "root" is in roots_to_fix, replace "root" with the correct dictionary base form. Leave all other pairs unchanged.

STRIP ONLY grammatical suffixes, and ONLY when the remainder is a valid Tamil dictionary word. Do NOT strip when the ending is part of the word (e.g. மகள் stays மகள்; never output ம for மகள்). Suffixes: கள், ஐ, இல், க்கு, ஆல், உடன், என்று, ஆக, ஆம், ஒடு, ஓடு, அது, வாறு; sandhi ச், ப், த் when suffix; ற்றுள், களுள் when suffix.

CRITICAL — NOT A SINGLE FORM MUST BE MISSED: Do not add or remove any entry. Every "form" must appear exactly once. Total pairs in output must equal total pairs in input. Words are very important.

OUTPUT (this exact JSON only):
{"normalized": [{"root": "root_word", "form": "original_form"}, ...]}
Return ONLY valid JSON, no other text."""


@tool
def fix_roots_in_list(normalized: list[dict], roots_to_fix: list[str]) -> dict[str, Any]:
    """Correct the specified roots in the normalized list to proper dictionary root form.

    Args:
        normalized: Full list of {"root": "...", "form": "..."} from normalize_to_root.
        roots_to_fix: List of root strings that are still inflected (from validate_root_forms).

    Returns:
        Tool result with corrected "normalized" list (same length, all forms preserved).
    """
    if not normalized or not roots_to_fix:
        return {"status": "success", "content": [{"text": json.dumps({"normalized": normalized}, ensure_ascii=False)}]}
    user_content = "Correct only these roots in the list: " + json.dumps(roots_to_fix, ensure_ascii=False) + "\nFull list:\n" + json.dumps(
        {"normalized": normalized}, ensure_ascii=False
    )
    out = _call_llm_json(FIX_ROOTS_SYSTEM, user_content)
    try:
        data = json.loads(out)
        if "normalized" not in data:
            data = {"normalized": normalized}
    except json.JSONDecodeError:
        data = {"normalized": normalized, "raw": out[:300]}
    return {"status": "success", "content": [{"text": json.dumps(data, ensure_ascii=False)}]}


class _GeminiModelNoEmptyTools(GeminiModel):
    """Omit tools key when no tools, so Gemini API does not reject the request."""

    def _format_request_config(
        self,
        tool_specs: list[Any] | None,
        system_prompt: str | None,
        params: dict[str, Any] | None,
    ) -> genai.types.GenerateContentConfig:
        config_kw: dict[str, Any] = {
            "system_instruction": system_prompt,
            **(params or {}),
        }
        if tool_specs:
            formatted_tools = self._format_request_tools(tool_specs)
            if formatted_tools:
                config_kw["tools"] = formatted_tools
        return genai.types.GenerateContentConfig(**config_kw)


model = _GeminiModelNoEmptyTools(
    client_args={"api_key": os.getenv("GEMINI_API_KEY")},
    model_id="gemini-2.0-flash",
    params={"temperature": 0},
)

# --- Prompts (same as main.py: noise_filter, root_normalizer, variant_grouping) ---
NOISE_FILTER_PROMPT = """You are the Noise Filter Agent. Your ONLY task is to remove three kinds of entries from a list of Tamil word candidates. Do nothing else.

Input: "candidates". Output: "filtered_candidates".


REMOVE ONLY THESE THREE (when clearly identified):
1.Meaningless words: non-lexical fragments that are not real dictionary words (e.g. standalone க்கு, ஆல் as fragments; லிளி; nonsensical OCR fragments). Keep real Tamil words even if short. When in doubt, KEEP.

2.Person names: first names, surnames, or full names of people (e.g. விஜய், கார்த்திக், கிறிஸ்டோபர், கொலம்பஸ், ஓடா). Keep common words that are not names (e.g. அரசன், மாணவன்).

3.Place names: cities, countries, states, regions (e.g. அமெரிக்கா, சென்னை, தமிழ்நாடு, இந்தியா). Keep common nouns (e.g. ஊர், நாடு when used as common words).

Process each candidate one by one from start to end. Apply the same rules to the last word as to the first. Do not miss or drop words in the middle or at the end of the list.

CRITICAL: Do NOT remove any other word. Do NOT miss any word. When in doubt, KEEP the word. Every candidate must appear in filtered_candidates unless it is clearly a person name, place name, or meaningless fragment.
WORDS IN CANDIDATE_LIST ARE VERY IMPORTANT AND SHOULD NOT BE REMOVED OR MISS WHILE RETURNING.


OUTPUT (this exact JSON only):
{
  "filtered_candidates": ["word1", "word2", "word3", "..."]
}
"""



# Root normalizer: tool1 → tool2 → if needs_fix then tool3; then output final normalized JSON.
ROOT_NORMALIZER_PROMPT = """You are the Root Normalizer Agent. You have three tools: normalize_to_root, validate_root_forms, fix_roots_in_list.

1. From the input, get "filtered_candidates" (list of Tamil words) from the previous node (noise_filter) or from "From noise_filter" section.
2. Call normalize_to_root with words= filtered_candidates. You get back {"normalized": [{"root": "root_word", "form": "original_form"}, ...]}.
3. Call validate_root_forms with normalized= that list. You get either:
   - {"status": "all_correct", "message": "ALL ARE IN CORRECT FORM"} → do NOT call fix_roots_in_list. Use the list from step 2 as the final normalized list.
   - {"status": "needs_fix", "roots_to_fix": ["root_a", "root_b", ...]} → call fix_roots_in_list with normalized= (list from step 2) and roots_to_fix= that list. Use the returned normalized list as the final list.
4. Output the final normalized list as your reply in this exact JSON format only (no other text):

{"normalized": [{"root": "root_word", "form": "original_form"}, ...]}

CRITICAL — NOT A SINGLE WORD MUST BE MISSED: Words are very very important. Every word from filtered_candidates MUST appear exactly once as "form" in your final output. The number of entries in "normalized" MUST equal the number of words in filtered_candidates. Do not drop, add, or merge any form. Only remove grammatical suffixes when the remaining part is a real dictionary word (e.g. மகள் → மகள், never ம).
"""

VARIANT_GROUPING_PROMPT = """You are the Variant Grouping Agent. Group grammatical variants under ONE root and output the final vocabulary. Input: "normalized" list of { "root", "form" }. Output: one JSON object (no wrapper key). Keys = Tamil root words. Values = arrays of ALL original "form" strings that belong to that root.

CRITICAL — NO FORM MAY BE MISSING: Every "form" in the input normalized list MUST appear in exactly one root's array. The union of all value arrays must equal the set of all "form" strings from the input. Do not omit any original form.

WHEN TO MERGE (same meaning, different grammar): அரசன், அரசர், அரசர்கள் → "அரசன்"; இரு, இருக்கும், இருந்தது → "இரு"; கடை, கடையில், கடைக்கு → "கடை".
WHEN TO KEEP SEPARATE (different meanings): பார் (see) vs பார்வை (vision); அரசு (government) vs அரசன் (king). Same spelling/root + same meaning → one key; list all forms under that key.

Return ONLY a single JSON object. Each key's array must list every original form that normalizes to that root. Count: total forms across all arrays = length of normalized list.

OUTPUT FORMAT:
{"root_word": ["form1", "form2", ...], ...}
"""

# --- Agents (3): noise_filter → root_normalizer (with tool) → variant_grouping ---
noise_filter_agent = Agent(
    name="noise_filter_agent",
    model=model,
    system_prompt=NOISE_FILTER_PROMPT,
    tools=[],
)
root_normalizer_agent = Agent(
    name="root_normalizer_agent",
    model=model,
    system_prompt=ROOT_NORMALIZER_PROMPT,
    tools=[normalize_to_root, validate_root_forms, fix_roots_in_list],
)
variant_grouping_agent = Agent(
    name="variant_grouping_agent",
    model=model,
    system_prompt=VARIANT_GROUPING_PROMPT,
    tools=[],
)

builder = GraphBuilder()
builder.add_node(noise_filter_agent, "noise_filter")
builder.add_node(root_normalizer_agent, "root_normalizer")
builder.add_node(variant_grouping_agent, "variant_grouping")
builder.add_edge("noise_filter", "root_normalizer")
builder.add_edge("root_normalizer", "variant_grouping")
builder.set_entry_point("noise_filter")
graph = builder.build()

NODE_ORDER = ["noise_filter", "root_normalizer", "variant_grouping"]


# --- Tamil letter (akshara) segmentation for variant-merge grouping ---
# Tamil Unicode: vowels 0B85-0B94, consonants 0B95-0BB9, virama 0BCD, vowel signs 0BBE-0BCC
_TAMIL_VIRAMA = "\u0BCD"
_TAMIL_VOWEL_SIGNS = (
    "\u0BBE\u0BBF\u0BC0\u0BC1\u0BC2\u0BC6\u0BC7\u0BC8\u0BCA\u0BCB\u0BCC"
)


def _tamil_letters(s: str) -> list[str]:
    """Segment Tamil string into letters (aksharas). Returns list of letter substrings."""
    if not s:
        return []
    letters = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        # Tamil independent vowel (0B85-0B94)
        if "\u0B85" <= c <= "\u0B94":
            letters.append(c)
            i += 1
            continue
        # Tamil consonant (0B95-0B99, 0B9A-0B9C, 0B9E, 0B9F-0BA3, 0BA8, 0BAA-0BB0, 0BB2-0BB9)
        if (
            "\u0B95" <= c <= "\u0B99"
            or "\u0B9A" <= c <= "\u0B9C"
            or c == "\u0B9E"
            or "\u0B9F" <= c <= "\u0BA3"
            or c == "\u0BA8"
            or "\u0BAA" <= c <= "\u0BB0"
            or "\u0BB2" <= c <= "\u0BB9"
        ):
            start = i
            i += 1
            if i < n and s[i] == _TAMIL_VIRAMA:
                i += 1  # consonant + virama = one letter (dead consonant)
            elif i < n and s[i] in _TAMIL_VOWEL_SIGNS:
                i += 1  # consonant + vowel sign = one letter
            letters.append(s[start:i])
            continue
        # Non-Tamil or other: treat as one "letter" so we don't break
        letters.append(c)
        i += 1
    return letters


def _tamil_letter_count(s: str) -> int:
    """Return number of Tamil letters (aksharas) in s."""
    return len(_tamil_letters(s))


def _tamil_prefix(s: str, n_letters: int) -> str:
    """Return prefix of s consisting of first n_letters Tamil letters."""
    letters = _tamil_letters(s)
    return "".join(letters[:n_letters]) if n_letters else ""


def _group_key_for_root(root: str) -> str:
    """Prefix key for grouping: by letter count, take first 1/2/3 letters."""
    n = _tamil_letter_count(root)
    if n <= 2:
        take = 1
    elif n == 3:
        take = 2
    else:
        take = 3
    return _tamil_prefix(root, take)


# --- Variant-merge validation (post-pipeline): merge same-meaning roots ---
VARIANT_MERGE_SYSTEM = """You are given a list of Tamil roots that share the same starting letters. Each root has its "forms" (original surface forms).

scenario : we already assign a task to covert the words to root form. But I doubt the system may be miss to convert them into root form, especially I doubt these words.
you are assigned to find root word for both and if they match keep them under one root otherwise keep them separate.

For each group: deside whether these roots have the same meaning or different meanings.

- If SAME meaning: return ONE root (the correct dictionary/canonical form) and combine ALL forms from all roots under that one root.
 Example: சொல்லு and சொல்லுதல்  → canonical root "சொல்லு", forms ["சொல்லுங்கள்", "சொல்லுதல்"].
- If DIFFERENT meanings: return each root unchanged with its own forms.
 Example: அண்ணன் (brother) vs அன்னம் (swan) → different.

CRITICAL: Do not drop any form. Total forms in output must equal total forms in input. If merging, use the more standard/canonical root (e.g. verb stem rather than verbal noun form when they mean the same).

Return ONLY valid JSON in this exact shape (no other text):
{"roots_with_forms": [{"root": "root_word", "forms": ["form1", "form2", ...]}, ...]}"""


def _variant_merge_call_llm(roots_with_forms: list[dict]) -> list[dict]:
    """Call LLM to resolve one doubt group. Returns list of {root, forms} (subset output)."""
    payload = {"roots_with_forms": roots_with_forms}
    user_content = json.dumps(payload, ensure_ascii=False)
    out = _call_llm_json(VARIANT_MERGE_SYSTEM, user_content)
    try:
        data = json.loads(out)
        items = data.get("roots_with_forms", data) if isinstance(data, dict) else data
        if isinstance(items, list) and items:
            return items
    except json.JSONDecodeError:
        pass
    return roots_with_forms  # fallback: no change


def validate_and_merge_duplicate_roots(vocab: dict[str, list[str]]) -> dict[str, list[str]]:
    """Post-pipeline step: group roots by Tamil letter prefix, ask LLM for doubt groups, merge same meaning. Returns new vocabulary (no forms dropped)."""
    if not vocab:
        return vocab
    # Group roots by prefix (by letter count)
    groups: dict[str, list[str]] = {}
    for root in vocab:
        key = _group_key_for_root(root)
        groups.setdefault(key, []).append(root)
    # Only groups with at least 2 roots
    result = dict(vocab)
    for key, roots in groups.items():
        if len(roots) < 2:
            continue
        roots_with_forms = [
            {"root": r, "forms": list(result[r])} for r in roots
        ]
        resolved = _variant_merge_call_llm(roots_with_forms)
        # Remove old roots for this group
        for r in roots:
            result.pop(r, None)
        # Add resolved roots (merge may combine into one or keep separate)
        for item in resolved:
            if not isinstance(item, dict) or "root" not in item:
                continue
            r = item["root"]
            forms = item.get("forms", [])
            if not isinstance(forms, list):
                forms = [forms] if forms else []
            result.setdefault(r, []).extend(forms)
        # Dedupe form lists for roots we just added
        for item in resolved:
            if isinstance(item, dict) and "root" in item:
                r = item["root"]
                if r in result:
                    result[r] = list(dict.fromkeys(result[r]))
    return result


def _get_agent_output(node_result):
    """Extract assistant text from a node's AgentResult."""
    if node_result is None or node_result.result is None:
        return ""
    msg = node_result.result.message
    if not msg or "content" not in msg:
        return ""
    for block in msg.get("content", []):
        if isinstance(block, dict) and "text" in block and block["text"]:
            return block["text"].strip()
    return ""


if __name__ == "__main__":
    import argparse
    from pathlib import Path as PathLib

    from word_candidates import ocr_text_to_word_candidates

    parser = argparse.ArgumentParser(
        description="Tamil vocabulary: OCR → word_candidates.py → noise_filter → root_normalizer (tool) → variant_grouping → vocabulary JSON."
    )
    parser.add_argument("--pdf", metavar="PATH", help="Get OCR from PDF (page by page)")
    parser.add_argument("--image", metavar="PATH", help="Get OCR from a single image")
    parser.add_argument(
        "-o", "--output-dir", default="pdf_output", help="Output dir (default: pdf_output)"
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PDF pages (default: 300)")
    args = parser.parse_args()

    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY is not set. Set it in .env or the environment.", file=sys.stderr)
        sys.exit(1)

    if args.pdf:
        from pdf_processor import get_ocr_per_page

        print("Getting OCR from PDF (non-agentic, page by page)...")
        pages_ocr = get_ocr_per_page(args.pdf, output_dir=args.output_dir, dpi=args.dpi)
        pdf_name = PathLib(args.pdf).stem
        combined_path = os.path.join(args.output_dir, f"{pdf_name}_full_ocr.txt")
        combined = "\n\n".join(f"--- Page {i} ---\n{t}" for i, t in enumerate(pages_ocr, start=1))
        os.makedirs(args.output_dir, exist_ok=True)
        with open(combined_path, "w", encoding="utf-8") as f:
            f.write(combined)
        print(f"✓ OCR saved to: {combined_path}")
    elif args.image:
        from ocr import tamil_ocr_from_image

        print("Getting OCR from image (non-agentic)...")
        pages_ocr = [tamil_ocr_from_image(args.image)]
    else:
        parser.error("Input required: use --pdf PATH or --image PATH.")

    input_basename = PathLib(args.pdf).stem if args.pdf else PathLib(args.image).stem
    os.makedirs(args.output_dir, exist_ok=True)

    header = "\n" + "=" * 60 + " PIPELINE OUTPUTS (word_candidates.py + 3 agents, root_normalizer with tool) " + "=" * 60
    page_vocabularies = []
    graph_failed = False

    for page_num, page_text in enumerate(pages_ocr, start=1):
        print(f"Page {page_num}/{len(pages_ocr)}: word_candidates.py → 3-agent graph...")
        # 1) Candidates from word_candidates.py (no agents)
        candidates = ocr_text_to_word_candidates(page_text, deduplicate=True)
        initial_input = json.dumps({"candidates": candidates}, ensure_ascii=False)

        try:
            result = graph(initial_input)
        except Exception as e:
            print(f"Error on page {page_num}: {e}", file=sys.stderr)
            graph_failed = True
            break

        pipeline_lines = [header]
        # Section: word_candidates (from word_candidates.py)
        word_candidates_block = f"\n--- word_candidates (word_candidates.py) ---\n{initial_input}\n"
        pipeline_lines.append(word_candidates_block)
        print(word_candidates_block)

        variant_grouping_json_text = None
        for node_id in NODE_ORDER:
            node_result = result.results.get(node_id)
            text = _get_agent_output(node_result)
            if text:
                if text.startswith("```"):
                    lines = text.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    text = "\n".join(lines)
                if node_id == "variant_grouping":
                    variant_grouping_json_text = text
            block = f"\n--- {node_id} ---\n{text}\n"
            print(block)
            pipeline_lines.append(block)

        page_path = os.path.join(
            args.output_dir, f"{input_basename}_page{page_num}_pipeline_output.txt"
        )
        with open(page_path, "w", encoding="utf-8") as f:
            f.write("\n".join(pipeline_lines))
        print(f"Pipeline output saved to: {page_path}")

        if variant_grouping_json_text:
            try:
                data = json.loads(variant_grouping_json_text)
                page_vocabularies.append(data)
            except json.JSONDecodeError:
                pass

    # Merge all page vocabularies
    merged = {}
    for vocab in page_vocabularies:
        if not isinstance(vocab, dict):
            continue
        for root, forms in vocab.items():
            if not isinstance(forms, list):
                continue
            merged.setdefault(root, []).extend(forms)
    for root in merged:
        merged[root] = list(dict.fromkeys(merged[root]))

    # Save old vocabulary (before variant-merge)
    old_vocab_path = os.path.join(args.output_dir, f"{input_basename}_old_vocabulary.json")
    with open(old_vocab_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Old vocabulary (before variant-merge) saved to: {old_vocab_path}")
    print(f"Old root count: {len(merged)}")

    # Post-pipeline: merge same-meaning roots (doubt groups by Tamil letter prefix)
    print("Running variant-merge validation (doubt groups by Tamil letter prefix)...")
    merged = validate_and_merge_duplicate_roots(merged)

    # Save final merged vocabulary (after variant-merge)
    final_merged_path = os.path.join(args.output_dir, f"{input_basename}_final_merged_vocabulary.json")
    with open(final_merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Final merged vocabulary (after variant-merge) saved to: {final_merged_path}")
    print(f"Final root count: {len(merged)}")

    # Also write to vocabulary.json for backward compatibility
    output_json_path = os.path.join(args.output_dir, f"{input_basename}_vocabulary.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Vocabulary (same as final) also saved to: {output_json_path}")

    if graph_failed:
        sys.exit(1)
