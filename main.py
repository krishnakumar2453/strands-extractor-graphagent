# Tamil vocabulary extraction pipeline — 5-agent graph
# Input: OCR text (no OCR agent; text provided directly)
from strands.multiagent import GraphBuilder
from strands import Agent
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv
from google import genai
from strands.models.gemini import GeminiModel

load_dotenv()


class _GeminiModelNoEmptyTools(GeminiModel):
    """Subclass that omits the tools key when there are no tools, so Gemini API does not reject the request (400)."""

    def _format_request_config(
        self,
        tool_specs: list[Any] | None,
        system_prompt: str | None,
        params: dict[str, Any] | None,
    ) -> genai.types.GenerateContentConfig:
        formatted_tools = self._format_request_tools(tool_specs)
        config_kw: dict[str, Any] = {
            "system_instruction": system_prompt,
            **(params or {}),
        }
        if formatted_tools:
            config_kw["tools"] = formatted_tools
        return genai.types.GenerateContentConfig(**config_kw)


model = _GeminiModelNoEmptyTools(
    client_args={"api_key": os.getenv("GEMINI_API_KEY")},
    model_id="gemini-2.0-flash",
    params={
        "temperature": 0,
    },
)

# --- (1) Word Candidate Agent (Extractor Phase 1: Identify ALL candidates) ---
WORD_CANDIDATE_PROMPT = """You are a Tamil language expert. Your job is to - Extract all Tamil words exactly as they appear in the OCR text.


INPUT: You receive the raw OCR text (a textbook page excerpt as a single string).
READING METHOD:
- Read line by line, word by word. Do NOT skip any section.
- OCR text is the primary source. NEVER invent words. NEVER reduce to root or group.

TYPE 1 - EXERCISES/ACTIVITIES (பயிற்சிகள்):
- IGNORE words with blanks/dashes (e.g. சன்_ல், த__காளி).
- IGNORE words inside boxes/tables/diagrams that are activity prompts (e.g. "வட்டமிடு", "சொல் உறுவாக்குவோம்").
- DO EXTRACT: headings, subheadings, and instructions that are outside fill-up/activity areas.

TYPE 2 - ALL OTHER SECTIONS:
- Extract EVERY Tamil word.
- Only meaningful words (2+ characters). SKIP single Tamil letters (அ, ஆ, க, ங, etc.).
- Do NOT guess or complete incomplete words.

Output a flat list of every Tamil word candidate you see. Recall over precision: if in doubt, include the token.

OUTPUT (this exact JSON structure only):
{
  "candidates": ["word1", "word2", "word3", "..."],
  "raw_text": "the full OCR Tamil text exactly as received"
}
"""

# --- (2) Word Candidate Validator Agent (TASK 1 only: find missing words) ---
WORD_CANDIDATE_VALIDATOR_PROMPT = """You are reviewing the Word Candidate agent output.
 Your ONLY task is to find ANY missing valid Tamil words from the OCR that were not extracted in the candidates list.

INPUT: You receive (1) "raw_text" and (2) "candidates" from the Word Candidate agent output.

TASK - FIND MISSING WORDS (HIGH PRIORITY):
1. Compare the OCR text (raw_text) against the extracted candidates and find ANY missing valid Tamil words.
2. Look very careful, check line by line.
3. if a valid missing word is found add that to the candidates list and return the list.
4. Output a single merged list with no duplicates.

YOU MUST NOT INCLUDE AS MISSING WORDS:
- Single characters (e.g. அ, ஆ, க, ங, ள, ளை, ழ, ற).
- Meaningless words (e.g. க்கு, ஆல், லிளி)
- Incomplete words with blanks or dashes (e.g. த__காளி, சன்_ல்).
- People names (e.g. விஜய், கார்த்திக்).
- City, Country, Place names (e.g. சென்னை, தமிழ்நாடு).
- Punctuation, symbols OR Numbers (e.g. 12.3, 1, 2).
- English  or other language text.
- Labels inside fill-up boxes, tables, or activity frames (e.g. வட்டமிடு, சொல் உறுவாக்குவோம் when used as box instructions).


IF NOTHING IS MISSING RETURN THE ORIGINAL CANDIDATES LIST UNCHANGED.
IF MISSING ADD MISSING WORDS TO CANDIDATES LIST .

**IMPORTANT : YOU SHOULD ONLY ABLE TO ADD MISSING WORDS TO THE CANDIDATES LIST.NOT TO REMOVE, CHANGE, OR MISS WHILE RETURNING THE CANDIDATES LIST**

OUTPUT (this exact JSON structure only):
{
  "candidates": ["word1", "word2", "word3", "..."]
}
"""

# --- (3) Noise Filter Agent (Extractor Phase 2: Apply discard rules) ---
NOISE_FILTER_PROMPT = """You are the Noise Filter Agent. Your ONLY task is to remove obvious non-lexical noise and unwanted entries from a list of Tamil word candidates ONLY IF PRESENT.

Input: "candidates". Output: "filtered_candidates" — only authentic Tamil words.


REMOVE ONLY THESE 5 CATEGORIES (if present). Do NOT remove anything else:

1. Blanks, placeholders, or fill-ups (e.g. ____, ___ல், த__காளி, சன்_ல்).
2. Proper nouns and name-like noise:
   - Person names (e.g. விஜய், கார்த்திக், கிறிஸ்டோபர், கொலம்பஸ் ).
   - Place, country, and city names (e.g. அமெரிக்கா, சென்னை, தமிழ்நாடு).
   - Writer or author names, and author initials (e.g. ஓடா, or single-letter/byline initials).
3. Single Tamil characters (e.g. அ, ஆ, க, ள, ங).
4. Pure numbers, symbols, or punctuation.
5. Table/diagram artifacts (labels or fragments that are not real words).

YOU ARE NOT ALLOWED TO REMOVE OTHER THAT THE ABOVE 5 CATEGORIES.
WORDS IN CANDIDATE_LIST ARE VERY IMPORTANT AND SHOULD NOT BE REMOVED OR MISS WHILE RETURNING.

OUTPUT (this exact JSON only):
{
  "filtered_candidates": ["word1", "word2", "word3", "..."]
}
"""
# --- (4) Root Normalizer Agent (Extractor Phase 3: Find root words) ---
ROOT_NORMALIZER_PROMPT = """You are the Root Normalizer Agent. Convert each Tamil word to its dictionary base/root form.
 Input: "filtered_candidates". Output: "normalized" list of { "root", "form" }. One word → one root. NEVER delete a word. NEVER merge different meanings.

 
ROOT WORD RULES:
- Strip all grammatical suffixes and return only the base form. Suffixes include: கள், ஐ, இல், க்கு, ஆல், உடன், என்று, ஆக, ஆம் , ஒடு/ஓடு, அது ,வாறு, ஆன, ஆனது, பட்டது,ள்ளது,கின்றனர் etc,.
 Root must never be the inflected form (e.g. இவ்வாறாகப் → root இவ்வாறு; பதினெட்டாம் → பதினெட்டு; வகுப்பில் → வகுப்பு; ஓரமாக → ஓரம்; மொழியொடு → மொழி; சிறந்தது → சிறந்த; அறியாதவாறு → அறியாத;மென்மையான → மென்மை; மென்மையானது → மென்மை; சேர்க்கப்பட்டது → சேர் ;சேர்த்துள்ளது → சேர் ).
- NOUNS → singular nominative (e.g. அரசன், மணம், நிறம்).
- VERBS → verb stem (e.g. செல், பார், இரு).
- Passive/compound verb forms (e.g. அழைக்கப்படுகிறார், கொண்டிருந்தான் ) → root = verb stem only: strip -க்கப்படு, -யப்பட்டு, -ப்பட்டிருந்தன, -ப்படுகிறார், etc. and return the stem (அழைக்கப்படு, கொண்டிரு).
- ADJECTIVES → base form (e.g. புதிய).

IMPORTANT : YOU ARE NOT ALLOWED TO REMOVE ANY WORD FROM THE CANDIDATES LIST. 
WORDS IN Filtered candidates are very important so do not remove ,miss any entry while returning 

COMPLETENESS: Output exactly one { "root", "form" } entry for every word in filtered_candidates. The length of "normalized" must equal the length of filtered_candidates. 
Include every form as a separate entry even when it differs only slightly from its root (e.g. one suffix or a minor spelling change). Do not omit a form because it resembles the root or duplicates the root string—each candidate word must appear exactly once as "form" with its root.

OUTPUT (this exact JSON only):
{
  "normalized": [
    { "root": "root_word", "form": "original_form" },
    ...
  ]
}
 
 
"""
# --- (5) Variant Grouping Agent (merge same roots, list all forms) ---
VARIANT_GROUPING_PROMPT = """You are the Variant Grouping Agent. Your ONLY task is to merge entries by root and list all forms.

Input: "normalized" — a list of objects, each with "root" and "form" (from the Root Normalizer).
Task: For each unique "root" value, collect ALL "form" values that have that root. One root → one key; value = list of every form that had that root. Do not group by meaning or similarity. Do not merge different roots. Same root string → merge into one list; different root string → separate keys.

Rules:
- Each key in the output is a root that appeared in the normalized list.
- Each value is the list of all "form" values whose "root" was that key.
- If the same (root, form) appears more than once, include that form only once in the list.
- Do not add or remove roots; do not combine two different roots into one. The normalizer already assigned the root; you only aggregate by that root.

Example:
Input normalized: [ {"root": "அரசன்", "form": "அரசன்"}, {"root": "அரசன்", "form": "அரசர்கள்"}, {"root": "கடை", "form": "கடை"} ]
Output: { "அரசன்": ["அரசன்", "அரசர்கள்"], "கடை": ["கடை"] }

Return ONLY a single JSON object. Keys = Tamil root words. Values = arrays of form strings for that root.

OUTPUT FORMAT:
{
  "root_word": ["form_1", "form_2", ...],
  ...
}
"""


# Create agents (all use same model; tools=[] so Gemini receives no tools)
word_candidate_agent = Agent(
    name="word_candidate_agent",
    model=model,
    system_prompt=WORD_CANDIDATE_PROMPT,
    tools=[],
)
word_candidate_validator_agent = Agent(
    name="word_candidate_validator_agent",
    model=model,
    system_prompt=WORD_CANDIDATE_VALIDATOR_PROMPT,
    tools=[],
)
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
    tools=[],
)
variant_grouping_agent = Agent(
    name="variant_grouping_agent",
    model=model,
    system_prompt=VARIANT_GROUPING_PROMPT,
    tools=[],
)
# Build the graph
builder = GraphBuilder()
builder.add_node(word_candidate_agent, "word_candidate")
builder.add_node(word_candidate_validator_agent, "word_candidate_validator")
builder.add_node(noise_filter_agent, "noise_filter")
builder.add_node(root_normalizer_agent, "root_normalizer")
builder.add_node(variant_grouping_agent, "variant_grouping")
builder.add_edge("word_candidate", "word_candidate_validator")
builder.add_edge("word_candidate_validator", "noise_filter")
builder.add_edge("noise_filter", "root_normalizer")
builder.add_edge("root_normalizer", "variant_grouping")
builder.set_entry_point("word_candidate")
graph = builder.build()

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


# Run with OCR text as input (no OCR agent; you provide the text or get it from PDF/image)
if __name__ == "__main__":
    import argparse
    from pathlib import Path as PathLib
    parser = argparse.ArgumentParser(description="Tamil vocabulary extraction: OCR text → 5-agent graph → vocabulary JSON.")
    parser.add_argument("--pdf", metavar="PATH", help="Get OCR from PDF (non-agentic: PDF → images → Gemini OCR)")
    parser.add_argument("--image", metavar="PATH", help="Get OCR from a single image (non-agentic)")
    parser.add_argument("-o", "--output-dir", default="pdf_output", help="Output dir for pipeline and vocabulary (default: pdf_output)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PDF pages (default: 300)")
    args = parser.parse_args()

    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY is not set. Set it in .env or the environment.", file=sys.stderr)
        sys.exit(1)

    # Build list of OCR texts: one per page (PDF = multiple pages, image = one page)
    if args.pdf:
        from pdf_processor import get_ocr_per_page
        print("Getting OCR from PDF (non-agentic, page by page)...")
        pages_ocr = get_ocr_per_page(args.pdf, output_dir=args.output_dir, dpi=args.dpi)
        # Save combined OCR for reference
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
        parser.error("Input required: use --pdf PATH or --image PATH to provide a PDF or image for OCR.")

    input_basename = PathLib(args.pdf).stem if args.pdf else PathLib(args.image).stem

    os.makedirs(args.output_dir, exist_ok=True)
    node_order = [
        "word_candidate",
        "word_candidate_validator",
        "noise_filter",
        "root_normalizer",
        "variant_grouping",
    ]
    header = "\n" + "=" * 60 + " PIPELINE OUTPUTS " + "=" * 60
    page_vocabularies = []
    graph_failed = False

    for page_num, page_text in enumerate(pages_ocr, start=1):
        print(f"Running 5-agent graph for page {page_num}/{len(pages_ocr)}...")
        try:
            result = graph(page_text)
        except Exception as e:
            print(f"Error on page {page_num}: {e}", file=sys.stderr)
            graph_failed = True
            break
        pipeline_lines = [header]
        variant_grouping_json_text = None
        for node_id in node_order:
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
        page_path = os.path.join(args.output_dir, f"{input_basename}_page{page_num}_pipeline_output.txt")
        with open(page_path, "w", encoding="utf-8") as f:
            f.write("\n".join(pipeline_lines))
        print(f"Pipeline output saved to: {page_path}")
        if variant_grouping_json_text:
            try:
                data = json.loads(variant_grouping_json_text)
                page_vocabularies.append(data)
            except json.JSONDecodeError:
                pass

    # Merge all page vocabularies: same root → union of variant lists, no duplicates
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
    root_count = len(merged)
    print(f"Final vocabulary (merged, deduped) saved to: {output_json_path}")
    print(f"Root words count: {root_count}")

    if graph_failed:
        sys.exit(1)
