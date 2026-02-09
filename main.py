# Tamil vocabulary extraction pipeline — 7-agent graph
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

# --- (1) Orchestrator ---
ORCHESTRATOR_PROMPT = """You are the Orchestrator for a Tamil vocabulary extraction pipeline.

INPUT: You receive raw OCR text from a Tamil textbook (page or excerpt). This is the OCR TEXT FOR REFERENCE for the rest of the pipeline.

RESPONSIBILITIES:
- Assign or confirm a page_id (e.g. "page_1" or from context).
- Pass the raw_text unchanged. Do not correct, guess, or delete any text.

RULES:
- Do NOT modify the OCR text. Do NOT remove or add words.
- Output only the structured message for the next agent.

OUTPUT (respond with this exact JSON structure only, no other text):
{
  "page_id": "string (e.g. page_1)",
  "raw_text": "the full OCR Tamil text exactly as received"
}
"""

# --- (2) Word Candidate Agent (Extractor Phase 1: Identify ALL candidates) ---
WORD_CANDIDATE_PROMPT = """You are a Tamil language expert. Your job is to identify ALL Tamil word candidates from the OCR text (Phase 1 only: extraction, no discard or root reduction).

INPUT: You receive page_id and raw_text (Tamil OCR from a textbook page). Image may be referenced for verification only.

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

# --- (5) Variant Grouping Agent (Extractor Phase 3: Group variants) ---
VARIANT_GROUPING_PROMPT = """You are the Variant Grouping Agent. Group grammatical variants under ONE root. Input: "normalized" list. Output: "groups" — root → list of variants.

CRITICAL: ONE root per distinct meaning. Merge all grammatical variants.

WHEN TO MERGE (same meaning, different grammar):
✓ அரசன், அரசர், அரசர்கள் → "அரசன்"
✓ இரு, இருக்கும், இருப்பது, இருந்தது → "இரு"
✓ கடை, கடையில், கடைக்கு → "கடை"

WHEN TO KEEP SEPARATE (different meanings):
✗ பார் (to see) vs பார்வை (vision/sight)
✗ அரசு (government/kingdom) vs அரசன் (king)

Test: If two roots feel repetitive/redundant, MERGE. Only keep separate if the meaning difference is significant and useful.

OUTPUT (this exact JSON only):
{
  "groups": {
    "கடை": ["கடை", "கடையில்", "கடைக்கு"],
    "உழவு": ["உழவு", "உழவுத்தொழில்"],
    "அரசன்": ["அரசன்", "அரசர்கள்"]
  }
}
"""

# --- (6) Coverage Auditor Agent (Validator: missing words + root corrections) ---
COVERAGE_AUDITOR_PROMPT = """You are the Coverage Auditor (validator). You do NOT remove or change data; you only report issues. Input: OCR raw_text + "groups" from Variant Grouping. Output: missing_words and root_corrections.

TASK 1 - FIND MISSING WORDS (compare OCR vs extracted groups):
- Find ANY valid Tamil word in the OCR that is not represented in the extracted groups.
- DO NOT include: single characters (அ, க, ளை); incomplete words with blanks (த__காளி); names, places, numbers, English; fill-up/box labels.

TASK 2 - ROOT CORRECTIONS (incorrect root forms / grouping errors):
- Report wrong root forms: e.g. if "நல்ல" should be "நல்லது" or vice versa, or if a verb stem is wrong.
- Report grouping errors: e.g. அரசு grouped with அரசன் (different meanings) or roots that should be merged but are separate.

Use empty objects if nothing to report.

OUTPUT (this exact JSON only):
{
  "missing_words": {
    "root": ["variant1", "variant2"]
  },
  "root_corrections": {
    "incorrect_root": "correct_root"
  },
  "suspicious_groups": [
    { "root": "அரசன்", "issue": "possible merge with அரசு" }
  ]
}
"""

# --- (7) Final Assembler Agent (Format as JSON, apply validator feedback) ---
FINAL_ASSEMBLER_PROMPT = """You are the Final Assembler. Produce the final vocabulary JSON. Input: "groups" from Variant Grouping + "missing_words", "root_corrections", "suspicious_groups" from Coverage Auditor.

RULES:
- NEVER delete words. NEVER drop entries. Add any "missing_words" from the auditor into the vocabulary.
- Apply "root_corrections": use correct_root where auditor said incorrect_root; merge variants into the correct root.
- Merge duplicates. Output format: root → { "variants": [...] } only. Do NOT include any explanations 

OUTPUT (this exact JSON only):
{
  "vocabulary": {
    "அரசன்": {
      "variants": ["அரசன்", "அரசர்கள்"]
    },
    "உழவு": {
      "variants": ["உழவு", "உழவுத்தொழில்"]
    }
  }
}
"""

# Create agents (all use same model; tools=[] so Gemini receives no tools)
orchestrator = Agent(
    name="orchestrator",
    model=model,
    system_prompt=ORCHESTRATOR_PROMPT,
    tools=[],
)
word_candidate_agent = Agent(
    name="word_candidate_agent",
    model=model,
    system_prompt=WORD_CANDIDATE_PROMPT,
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
coverage_auditor_agent = Agent(
    name="coverage_auditor_agent",
    model=model,
    system_prompt=COVERAGE_AUDITOR_PROMPT,
    tools=[],
)
final_assembler_agent = Agent(
    name="final_assembler_agent",
    model=model,
    system_prompt=FINAL_ASSEMBLER_PROMPT,
    tools=[],
)

# Build the graph
builder = GraphBuilder()

builder.add_node(orchestrator, "orchestrator")
builder.add_node(word_candidate_agent, "word_candidate")
builder.add_node(noise_filter_agent, "noise_filter")
builder.add_node(root_normalizer_agent, "root_normalizer")
builder.add_node(variant_grouping_agent, "variant_grouping")
builder.add_node(coverage_auditor_agent, "coverage_auditor")
builder.add_node(final_assembler_agent, "final_assembler")

# Linear pipeline: orchestrator → word_candidate → noise_filter → ...
builder.add_edge("orchestrator", "word_candidate")
builder.add_edge("word_candidate", "noise_filter")
builder.add_edge("noise_filter", "root_normalizer")
builder.add_edge("root_normalizer", "variant_grouping")
builder.add_edge("variant_grouping", "coverage_auditor")
builder.add_edge("coverage_auditor", "final_assembler")

builder.set_entry_point("orchestrator")
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


# Run with OCR text as input (no OCR agent; you provide the text)
if __name__ == "__main__":
    sample_ocr = """
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
    result = graph(sample_ocr.strip())

    # Print each agent's output only (in pipeline order)
    node_order = [
        "orchestrator",
        "word_candidate",
        "noise_filter",
        "root_normalizer",
        "variant_grouping",
        "coverage_auditor",
        "final_assembler",
    ]
    header = "\n" + "=" * 60 + " PIPELINE OUTPUTS " + "=" * 60
    print(header)
    pipeline_lines = [header]
    final_assembler_json_text = None
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
            if node_id == "final_assembler":
                final_assembler_json_text = text
        block = f"\n--- {node_id} ---\n{text}\n"
        print(block)
        pipeline_lines.append(block)

    # Save full pipeline output to a text file
    PIPELINE_OUTPUT_PATH = "pipeline_output.txt"
    with open(PIPELINE_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(pipeline_lines))
    print(f"Pipeline output saved to: {PIPELINE_OUTPUT_PATH}")

    # Save final_assembler output to a JSON file
    OUTPUT_JSON_PATH = "vocabulary_output.json"
    if final_assembler_json_text:
        try:
            data = json.loads(final_assembler_json_text)
            with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Final vocabulary saved to: {OUTPUT_JSON_PATH}")
        except json.JSONDecodeError as e:
            print(f"Could not parse final_assembler output as JSON: {e}")
            with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
                f.write(final_assembler_json_text)
            print(f"Raw output written to: {OUTPUT_JSON_PATH}")
