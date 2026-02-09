# Tamil vocabulary extraction pipeline — 5-agent graph
# Input: OCR text (no OCR agent; text provided directly)
from strands.multiagent import GraphBuilder
from strands import Agent
import json
import os
from dotenv import load_dotenv
from strands.models.gemini import GeminiModel

load_dotenv()

model = GeminiModel(
    client_args={"api_key": os.getenv("GEMINI_API_KEY")},
    model_id="gemini-2.0-flash",
    params={
        "temperature": 0.3
    },
)

# --- (1) Word Candidate Agent (Extractor Phase 1: Identify ALL candidates) ---
WORD_CANDIDATE_PROMPT = """You are a Tamil language expert. Your job is to identify ALL Tamil word candidates from the OCR text (Phase 1 only: extraction, no discard or root reduction).

INPUT: You receive the raw OCR text (the full Tamil textbook excerpt as a single string). Include that exact raw text in your output as "raw_text" so the next agent can use it.

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
  "candidates": ["விவசாயத்தில்", "உழவுத்தொழில்", "மக்கள்", "..."],
  "raw_text": "the full OCR Tamil text exactly as received"
}
"""

# --- (2) Word Candidate Validator Agent (TASK 1 only: find missing words) ---
WORD_CANDIDATE_VALIDATOR_PROMPT = """You are reviewing the Word Candidate agent output. Your ONLY task is to find ANY missing valid Tamil words from the OCR that were not extracted in the candidates list.

INPUT: You receive (1) "raw_text" and (2) "candidates" from the Word Candidate agent output.

TASK - FIND MISSING WORDS (HIGH PRIORITY):
1. Compare the OCR text against the extracted candidates and find ANY missing valid Tamil words.
2. Output a single merged list: original candidates plus any valid missing words, with no duplicates.

YOU MUST NOT INCLUDE AS MISSING WORDS:
- Single characters (e.g. அ, ஆ, க, ங, ள, ளை, ழ, ற).
- Incomplete words with blanks or dashes (e.g. த__காளி, சன்_ல்).
- People names (e.g. விஜய், கார்த்திக், மின்னி, தூரன்).
- Place names (e.g. சென்னை, தமிழ்நாடு).
- Numbers (e.g. 12.3, 1, 2).
- English text.
- Punctuation or symbols.
- Labels inside fill-up boxes, tables, or activity frames (e.g. வட்டமிடு, சொல் உறுவாக்குவோம் when used as box instructions).
- Any token that is not a valid, complete Tamil word (must be 2+ characters, no blanks).

If nothing is missing, return the original candidates list unchanged.

OUTPUT (this exact JSON structure only):
{
  "candidates": ["விவசாயத்தில்", "உழவுத்தொழில்", "மக்கள்", "..."]
}
"""

# --- (3) Noise Filter Agent (Extractor Phase 2: Apply discard rules) ---
NOISE_FILTER_PROMPT = """You are the Noise Filter Agent. Apply DISCARD RULES only. Input: "candidates". Output: "clean_words" — only authentic Tamil words.

DISCARD (remove): Single Tamil characters (அ, க, ள, etc.); uyirmei fragments that are not words; fill-up placeholders (____, சன்_ல், த__காளி); people names (விஜய், கார்த்திக்); place names (சென்னை, தமிழ்நாடு); numbers, punctuation, English; diagram/table artifacts; incomplete words. KEEP: valid Tamil words (2+ chars, no blanks). If unsure → KEEP.

OUTPUT (this exact JSON only):
{
  "clean_words": ["விவசாயத்தில்", "உழவுத்தொழில்", "மக்கள்"]
}
"""

# --- (4) Root Normalizer Agent (Extractor Phase 3: Find root words) ---
ROOT_NORMALIZER_PROMPT = """You are the Root Normalizer Agent. Convert each Tamil word to its dictionary base/root form. Input: "clean_words". Output: "normalized" list of { "root", "form" }. One word → one root. NEVER delete a word. NEVER merge different meanings.

ROOT WORD RULES:
- Remove suffixes: கள், ஐ, இல், க்கு, ஆல், உடன், என்று, etc. Return dictionary base form.
- NOUNS → singular nominative (e.g. அரசன், மணம், நிறம்).
- VERBS → verb stem (e.g. செல், பார், இரு).
- ADJECTIVES → base form (e.g. புதிய).
- Same root with different case markers → one root (e.g. நிலப்பகுதி, நிலப்பகுதியை, நிலப்பகுதியின் → root நிலப்பகுதி).

OUTPUT (this exact JSON only):
{
  "normalized": [
    { "root": "விவசாயம்", "form": "விவசாயத்தில்" },
    { "root": "உழவு", "form": "உழவுத்தொழில்" },
    { "root": "மக்கள்", "form": "மக்கள்" }
  ]
}
"""

# --- (5) Variant Grouping Agent (Group variants and output final vocabulary JSON) ---
VARIANT_GROUPING_PROMPT = """You are the Variant Grouping Agent. Group grammatical variants under ONE root and output the final vocabulary. Input: "normalized" list. Output: the final vocabulary JSON only (no wrapper key).

CRITICAL: ONE root per distinct meaning. Merge all grammatical variants. Do not miss any word.

WHEN TO MERGE (same meaning, different grammar):
✓ அரசன், அரசர், அரசர்கள் → "அரசன்"
✓ இரு, இருக்கும், இருப்பது, இருந்தது → "இரு"
✓ கடை, கடையில், கடைக்கு → "கடை"

WHEN TO KEEP SEPARATE (different meanings):
✗ பார் (to see) vs பார்வை (vision/sight)
✗ அரசு (government/kingdom) vs அரசன் (king)

Return ONLY a single JSON object. Keys are Tamil root words. Values are arrays of all surface forms (original forms) for that root.

OUTPUT FORMAT:
{
  "root_word": ["original_form_1", "original_form_2"],
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

# Pipeline: word_candidate → word_candidate_validator → noise_filter → root_normalizer → variant_grouping (final output)
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
    parser = argparse.ArgumentParser(description="Tamil vocabulary extraction: OCR text → 5-agent graph → vocabulary JSON.")
    parser.add_argument("--pdf", metavar="PATH", help="Get OCR from PDF (non-agentic: PDF → images → Gemini OCR)")
    parser.add_argument("--image", metavar="PATH", help="Get OCR from a single image (non-agentic)")
    parser.add_argument("-o", "--output-dir", default="pdf_output", help="Output dir for PDF OCR (default: pdf_output)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PDF pages (default: 300)")
    args = parser.parse_args()

    if args.pdf:
        from pdf_processor import process_pdf
        print("Getting OCR from PDF (non-agentic)...")
        ocr_text = process_pdf(args.pdf, output_dir=args.output_dir, dpi=args.dpi)
    elif args.image:
        from ocr import tamil_ocr_from_image
        print("Getting OCR from image (non-agentic)...")
        ocr_text = tamil_ocr_from_image(args.image)
    else:
        ocr_text = """
    --- Page 0 ---
12.3 பாடிமகிழ்வோம்

மின்னி மின்னி வானத்தில்

மின்னும் மின்மினி போலவே
மின்னி மின்னி வானத்தில்
என்னை உற்றுப் பார்த்திடும்
சின்னச் சின்ன மீன்களே

என்ன அங்கே செய்கின்றீர்?
எப்படித் தான் நிற்கிறீர்?

அன்னை கழுத்தில் அணிந்திடும்
அழகு வைர மாலையில்

மின்னும் மணிகள் போலவும்
வெள்ளை முத்தைப் போலவும்
பொன்னுப் பூவைப் போலவும்
மின்னிச் சிரிக்கும் மீன்களே

எட்ட நின்று பார்க்கிறீர்
இரவில் மேலே வருகிறீர்
பட்டப் பகலில் எங்குதான்
பறந்து செல்வீர் கூறுங்கள்

-பெ. தூரன்

--- Page 1 ---
12.4 தெரிந்துகொள்வோம்
ல, ழ, ள ஒலிவேறுபாடு

சிலம்பம் ஒரு கலை.

வயலில் களை பறித்தார்.

கூடைகள் செய்யக் கழை பயன்படுகிறது.

கடல் அலையில் விளையாடினான்.

பாம்பு அளையிலிருந்து வெளியே வந்தது.

அம்மா குழந்தையை அழைத்தார்.

--- Page 2 ---
எனக்கு முழங்காலில் வலி.

வளியால் மரங்கள் அசைந்தன.

இது நான் போகும் வழி.

கிளிக்குக் கூர்மையான அலகு உண்டு.

அளகு அடைகாக்கிறது.

வானுக்கு நிலவு அழகு.

--- Page 3 ---
பொருள் அறிவோம்

1. களை - தேவையற்ற செடிகள்
2. கழை - மூங்கில்
3. அளை - புற்று
4. அழை - கூப்பிடுதல்
5. வளி - காற்று
6. வழி - பாதை
7. அலகு - பறவையின் மூக்கு
8. அளகு - கோழி
பயிற்சி
பொருத்தமான எழுத்தை எடுத்து எழுதுக.
2
அ
ளை ழை
க
ளை
ழை
வ
லி
ளி
ழி
ള
ழ
அ
ளை
ழை

    """
    print("Running 5-agent graph on OCR text...")
    # Graph entry expects str | list[Contentblock] | Messages | None (not dict)
    result = graph(ocr_text.strip())

    # Print each agent's output only (in pipeline order)
    node_order = [
        "word_candidate",
        "word_candidate_validator",
        "noise_filter",
        "root_normalizer",
        "variant_grouping",
    ]
    header = "\n" + "=" * 60 + " PIPELINE OUTPUTS " + "=" * 60
    print(header)
    pipeline_lines = [header]
    variant_grouping_json_text = None
    for node_id in node_order:
        node_result = result.results.get(node_id)
        text = _get_agent_output(node_result)
        if text:
            # Strip markdown code fence for cleaner display / parsing
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

    # Save full pipeline output to a text file
    PIPELINE_OUTPUT_PATH = "pipeline_output.txt"
    with open(PIPELINE_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(pipeline_lines))
    print(f"Pipeline output saved to: {PIPELINE_OUTPUT_PATH}")

    # Save variant_grouping (final vocabulary) output to a JSON file
    OUTPUT_JSON_PATH = "vocabulary_output.json"
    if variant_grouping_json_text:
        try:
            data = json.loads(variant_grouping_json_text)
            with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Final vocabulary saved to: {OUTPUT_JSON_PATH}")
        except json.JSONDecodeError as e:
            print(f"Could not parse variant_grouping output as JSON: {e}")
            with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
                f.write(variant_grouping_json_text)
            print(f"Raw output written to: {OUTPUT_JSON_PATH}")
