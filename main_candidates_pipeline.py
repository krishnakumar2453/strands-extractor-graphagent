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
NORMALIZE_SYSTEM = """You convert Tamil words to their dictionary base/root form. Strip ONLY grammatical suffixes (case, number, tense). Do NOT strip when the ending is part of the word (e.g. மகள் → மகள்). Suffixes to strip when clearly attached: கள், ஐ, இல், க்கு, ஆல், உடன், என்று, ஆக, ஆம், ஒடு, ஓடு, அது, வாறு; word-final sandhi ச், ப், த் when sandhi. Root = dictionary form. Every input word MUST appear exactly once as "form" in the output. Return ONLY valid JSON: {"normalized": [{"root": "...", "form": "..."}, ...]}."""


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
VALIDATE_SYSTEM = """You validate a list of Tamil root/form pairs. For each pair, check if "root" is in correct dictionary root form (no grammatical suffixes). Grammatical suffixes include: கள், ஐ, இல், க்கு, ஆல், உடன், என்று, ஆக, ஆம், ஒடு, ஓடு, அது, வாறு, ற்றுள், களுள்; word-final sandhi ச், ப், த் when they are suffixes. Do NOT flag roots where the ending is part of the word (e.g. மகள் is correct).

If ALL roots are in correct form, return exactly: {"status": "all_correct", "message": "ALL ARE IN CORRECT FORM"}.
If ANY root still has a grammatical suffix, return: {"status": "needs_fix", "roots_to_fix": ["root1", "root2", ...]} listing only the root strings that need to be corrected. Return ONLY valid JSON."""


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
FIX_ROOTS_SYSTEM = """You correct only the specified roots in a normalized list. Input: full list of {"root", "form"} and a list of root strings that need fixing. For each pair whose "root" is in the roots_to_fix list, replace "root" with the correct dictionary base form (strip only grammatical suffixes; e.g. மகள் stays மகள்). Leave all other pairs unchanged. Do not add or remove any entry. Every "form" must still appear exactly once. Return ONLY valid JSON: {"normalized": [{"root": "...", "form": "..."}, ...]}."""


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
2. Call normalize_to_root with words= filtered_candidates. You get back {"normalized": [{"root": "...", "form": "..."}, ...]}.
3. Call validate_root_forms with normalized= that list. You get either:
   - {"status": "all_correct", "message": "ALL ARE IN CORRECT FORM"} → do NOT call fix_roots_in_list. Use the list from step 2 as the final normalized list.
   - {"status": "needs_fix", "roots_to_fix": ["root_a", "root_b", ...]} → call fix_roots_in_list with normalized= (list from step 2) and roots_to_fix= that list. Use the returned normalized list as the final list.
4. Output the final normalized list as your reply in this exact JSON format only (no other text): {"normalized": [{"root": "...", "form": "..."}, ...]}

CRITICAL: Every word from filtered_candidates MUST appear exactly once as "form" in your final output. Do not drop or add any form.
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

    output_json_path = os.path.join(args.output_dir, f"{input_basename}_vocabulary.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Final vocabulary (merged, deduped) saved to: {output_json_path}")
    print(f"Root words count: {len(merged)}")

    if graph_failed:
        sys.exit(1)
