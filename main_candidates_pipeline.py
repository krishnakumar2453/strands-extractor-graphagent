# Tamil vocabulary extraction pipeline — word_candidates.py + compound-word graph
# Word candidates from word_candidates.py. Then: noise_filter → lexical_router → [bridge: morphological_extractor on SPLIT + suffix filter + convert] → root_normalizer → variant_grouping → variant_grouping_validation.
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

# Model for all pipeline agents (noise_filter, lexical_router, morphological_extractor, root_normalizer, variant_grouping, variant_grouping_validation). OCR uses its own model in ocr.py.
AGENT_MODEL_ID = "gemini-2.5-pro"

# --- LLM helper for tools ---
def _call_llm_json(system: str, user_content: str) -> str:
    """Call Gemini with system + user content; return raw response text. Uses AGENT_MODEL_ID (Gemini Pro)."""
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
        model=AGENT_MODEL_ID,
        contents=user_content,
        config=config,
    )
    return (response.text or "").strip() or "{}"


# --- Suffix stop list: tokens that must not become vocabulary roots (GSR on classifier) ---
PLURALS = {"கள்", "களை", "மார்கள்"}

CASE_MARKERS = {
    "அ", "ஐ", "இ", "ந்த", "ஆல்", "ஆன்", "க்கு", "கு", "உக்கு", "இன்", "ஆது", "இல்",
    # Participles and missing Genitive
    "த்த", "ற்ற", "கின்ற", "க்க", "கிய", "கியத்", "அது"
}

ADVERBIAL_MARKERS = {
    "ஆகு", "ஆக", "ஆய்", "பது", "யது", "வது"
}

CLITICS = {"ஏ", "ஓ", "ஆ", "ஆவது"}

VERB_EXTENSIONS = {
    "பதற்கு", "வதற்கு", "ஆர்", "ஆர்கள்", "அனர்", "வாறு",
    "ஆள்", "ஓம்", "ஈர்கள்",
    # Present Tense Conjugations (கிறது / கின்றது family)
    "கிறது", "கின்றது", "கிறான்", "கின்றான்", "கிறாள்", "கின்றாள்",
    "கிறார்", "கின்றார்", "கிறார்கள்", "கின்றார்கள்", "கின்றனர்", "கின்றன"
}

SANDHI_VARIANTS = {
    "ஐக்", "ஐச்", "ஐத்", "ஐப்", "க்குச்", "க்குத்", "க்குப்", "க்க்",
    "களைக்", "களைச்", "களைத்", "களைப்", "ஆகக்", "ஆகச்", "ஆகத்", "ஆகப்",
    "அத்", "இத்",
    # Empty Morphemes (சாரியைகள்)
    "அற்று", "இற்று", "அன்"
}

# The 'Ae' (long E) series in Tamil (Emphatic fusion)
TAMIL_AE_SERIES = {
    "கே", "ஙே", "சே", "ஞே", "டே", "ணே",
    "தே", "நே", "பே", "யே", "ரே",
    "லே", "வே", "ழே", "ளே", "றே", "னே"
}

# The 'Aa' (long A) series in Tamil (Interrogative fusion)
TAMIL_AA_SERIES = {
    "கா", "ஙா", "சா", "ஞா", "டா", "ணா",
    "தா",  "யா", "ரா",
    "லா", "வா", "ழா", "ளா", "றா", "னா"
}

# The 'Oo' (long O) series in Tamil (Doubt/Emphasis fusion)
TAMIL_OO_SERIES = {
    "கோ", "ஙோ", "சோ", "ஞோ", "டோ", "ணோ",
    "தோ", "நோ", "போ", "மோ", "யோ", "ரோ",
    "லோ", "வோ", "ழோ", "ளோ", "றோ", "னோ"
}

# --- SAFETY NET: Valid single-letter dictionary roots that must NEVER be stripped ---
PROTECTED_ROOTS = {"கா", "தா", "கோ", "தே", "மே"}

# Combine suffix stop lists into one frozen set for fast O(1) lookup
MASTER_SUFFIX_STOP_LIST = frozenset(
    PLURALS | CASE_MARKERS | ADVERBIAL_MARKERS
    | CLITICS | VERB_EXTENSIONS | SANDHI_VARIANTS
    | TAMIL_AE_SERIES | TAMIL_AA_SERIES | TAMIL_OO_SERIES
)

SUFFIX_STOP_LIST = MASTER_SUFFIX_STOP_LIST

# Leading sandhi consonants to strip from start of root words (e.g. "ப்பை" -> "பை")
LEADING_SANDHI = frozenset({"க்", "ச்", "த்", "ப்"})


def _strip_leading_sandhi(s: str) -> str:
    """If s starts with a leading sandhi consonant (க், ச், த், ப்), remove it and return the rest. Otherwise return s unchanged."""
    s = (s or "").strip()
    if len(s) >= 2 and s[:2] in LEADING_SANDHI:
        return s[2:].strip()
    return s


def filter_suffixes_from_classifier_output(
    classifier_output: list, suffix_list: list[str] | frozenset[str] | None = None
) -> list[dict]:
    """Remove suffix-only tokens from root_words in SPLIT items. Preserve every form.
    If suffix_list is None, uses SUFFIX_STOP_LIST. Returns cleaned classifier output (same length)."""
    stop = frozenset(suffix_list) if suffix_list is not None else SUFFIX_STOP_LIST
    out: list[dict] = []
    for item in classifier_output or []:
        if not isinstance(item, dict):
            out.append(item)
            continue
        form = item.get("form") or ""
        decision = item.get("decision", "")
        if decision == "KEEP":
            out.append(dict(item))
            continue
        if decision == "SPLIT":
            roots = item.get("root_words") or []
            # Strip leading sandhi (க், ச், த், ப்) from each root, then filter by suffix stop list
            normalized = [_strip_leading_sandhi(r if isinstance(r, str) else str(r)) for r in roots]
            protected = PROTECTED_ROOTS if suffix_list is None else frozenset()
            filtered = [r for r in normalized if r and (r not in stop or r in protected)]
            if not filtered:
                out.append({"form": form, "decision": "KEEP", "root_word": form})
            else:
                out.append({"form": form, "decision": "SPLIT", "root_words": filtered})
            continue
        out.append(dict(item))
    return out


# --- Converter: classifier output → split_candidates (deterministic) ---
def classifier_output_to_split_candidates(classifier_output: list) -> list[dict]:
    """Convert classifier output to flat list of {root, form}. KEEP → one pair; SPLIT → one pair per root in root_words."""
    out: list[dict] = []
    for item in classifier_output or []:
        if not isinstance(item, dict):
            continue
        form = item.get("form") or ""
        decision = item.get("decision", "")
        if decision == "KEEP":
            root = (item.get("root_word") or "").strip()
            if form:
                out.append({"root": root, "form": form})
        elif decision == "SPLIT":
            roots = item.get("root_words") or []
            for r in roots:
                root = (r if isinstance(r, str) else str(r)).strip()
                if form:
                    out.append({"root": root, "form": form})
    return out


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
    model_id=AGENT_MODEL_ID,
    params={"temperature": 0, "max_output_tokens": 65536},
)

# --- Prompts: noise_filter, lexical_router, morphological_extractor, root_normalizer, variant_grouping ---
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




# --- Agent 1: Lexical Router (KEEP vs SPLIT only) ---
LEXICAL_ROUTER_PROMPT = """You are the Lexical Router Agent for a Tamil Dictionary Pipeline.
From the input node, you will receive "filtered_candidates" (a list of Tamil words).
Your ONLY job is to classify each word as either KEEP or SPLIT based on dictionary standards.

RULES:
- KEEP:
  (a) Standard dictionary base words (e.g., சிறுவன், அவர்).
  (b) Lexical compounds / closed nouns representing a single concept (e.g., பாடசாலை, இணையதளம், குச்சிமிட்டாய்).
- SPLIT:
  Agglutinative phrases, words with postpositions (போல், உடன், எல்லாம்), or inflected nouns/verbs that contain suffixes for tense, case, or plural markers.

OUTPUT FORMAT:
Return ONLY a JSON array. Do not include markdown wrappers or explanations.
- If KEEP: {"form": "<surface form>", "decision": "KEEP", "root_word": "<dictionary entry>"}
- If SPLIT: {"form": "<surface form>", "decision": "SPLIT"}

EXAMPLES:
[
 {"form": "இணையதளம்", "decision": "KEEP"},
 {"form": "இணையதளம்", "decision": "KEEP"},
 {"form": "அச்சிறுவனைப்", "decision": "SPLIT"},
 {"form": "செய்துகொண்டிருக்கின்றனர்", "decision": "SPLIT"}
]
"""

# --- Agent 2: Morphological Extractor (SPLIT words only: root_words + suffixes) ---
MORPHOLOGICAL_EXTRACTOR_PROMPT = """You are the Tamil Morphological Parser Agent (பகுபத உறுப்பிலக்கணம்).
You will receive a list of Tamil words classified as "SPLIT" (input: JSON with key "split_forms").
Your ONLY job is to accurately extract the base roots (பகுதி) and meaningful grammatical suffixes (இடைநிலை, சாரியை, விகுதி).

EXTRACTION RULES:
1. `root_words`: Extract ONLY valid, standalone dictionary base words.
   - Do NOT over-split base nouns or pronouns (e.g., Use "சிறுவன்", NOT "சிறு" + "அன்". Use "அவர்", NOT "அ" + "அர்").
   - Postpositions (e.g., போல், போல, உடன், எல்லாம், பற்றி) MUST be treated as independent roots in the `root_words` array.
   - For compound verbs, extract all base verb roots and reverse mutations (e.g., "கொண்டிருந்தவர்" -> ["கொள்", "இரு"]).
2. `suffixes`: Extract meaningful grammatical markers (e.g., கள், ஐ, ஆல், கு, இன், அது, கண், இல், கின்று, கிறு, ஆன், ஆர், அர், ஓம், ஏ, உம், ஆக, ஆன).
   - Do NOT drop the final terminal suffix of a word.
3. STRICT EXCLUSION (சந்தி/விகாரம்):
   - Completely discard single-consonant phonetic links (க், ச், ட், த், ப், ற், ந், வ், ய்). NEVER place them in the suffixes array.

OUTPUT FORMAT:
Return ONLY a JSON array. Do not include markdown wrappers or explanations.
{"form": "<surface form>", "decision": "SPLIT", "root_words": ["<root1>", "<root2>"], "suffixes": ["<suffix1>", "<suffix2>"]}

EXAMPLES:
[
 {"form": "தம்மைப்போலவே", "decision": "SPLIT", "root_words": ["தாம்", "போல்"], "suffixes": ["ஐ", "ஏ"]},
 {"form": "கொண்டிருந்தவர்", "decision": "SPLIT", "root_words": ["கொள்", "இரு"], "suffixes": ["அ", "அவர்"]},
 {"form": "உதவிகளைச்", "decision": "SPLIT", "root_words": ["உதவி"], "suffixes": ["கள்", "ஐ"]}
]
"""

# --- Root normalizer: one agent that outputs canonical dictionary roots (no tools) ---
ROOT_NORMALIZER_PROMPT = """You are the Root Normalizer. Your only job is to turn the input list of root/form pairs into the same list with every "root" in canonical Tamil dictionary base form.

Input: JSON with key "split_candidates" — an array of {"root", "form"}. Each "form" is the original surface string; each "root" may still carry inflection or case.

Output: exactly one JSON object with key "normalized" — an array of {"root", "form"} with the same length and the same order. For each entry:
- "form" is unchanged.
- "root" is the dictionary base form: the canonical form you would look up in a Tamil dictionary. Remove any grammatical inflection or case ending from the root so it is a proper lemma. Do not change a root if the result of stripping would not be a valid Tamil word (e.g. leave மகள் as மகள்).

Rules: same number of entries as input; every "form" preserved; only "root" may change. Reply with nothing but the JSON object (no markdown, no explanation)."""

VARIANT_GROUPING_PROMPT = """You are the Variant Grouping Agent. From the previous node (root_normalizer) get the list under key "normalized" (array of { "root", "form" }). Group grammatical variants under ONE root and output the final vocabulary. Output: one JSON object (no wrapper key). Keys = Tamil root words. Values = arrays of ALL original "form" strings that belong to that root.

CRITICAL — NO FORM MAY BE MISSING: Every "form" in the input normalized list MUST appear in exactly one root's array. The union of all value arrays must equal the set of all "form" strings from the input. Do not omit any original form.

WHEN TO MERGE (same meaning, different grammar): அரசன், அரசர், அரசர்கள் → "அரசன்"; இரு, இருக்கும், இருந்தது → "இரு"; கடை, கடையில், கடைக்கு → "கடை".
WHEN TO KEEP SEPARATE (different meanings): பார் (see) vs பார்வை (vision); அரசு (government) vs அரசன் (king). Same spelling/root + same meaning → one key; list all forms under that key.

Return ONLY a single JSON object. Each key's array must list every original form that normalizes to that root. Count: total forms across all arrays = length of normalized list.

OUTPUT FORMAT:
{"root_word": ["form1", "form2", ...], ...}
"""

VARIANT_GROUPING_VALIDATION_PROMPT = """
### ROLE
You are the Variant Grouping Validation Agent. Your purpose is to normalize a Tamil vocabulary JSON by merging redundant roots and ensuring all keys are canonical dictionary forms.

### INPUT SPECIFICATION
A JSON object where:
- Keys = Potential Tamil roots (may contain errors or inflections).
- Values = Arrays of surface forms (strings).

### STRICT TASK RULES
1. Root Normalization (De-inflection): Every key MUST be a proper Tamil dictionary root.
   - Remove grammatical suffixes from keys (e.g., Change "வகுப்பில்" to "வகுப்பு", "சொல்லுதல்" to "சொல்லு").
   - Do not strip a suffix if the remainder would not be a valid word (e.g. keep மகள் as மகள்).
2. Deduplication & Merging: If two keys represent the same semantic root (e.g., "சொல்லு" and "சொல்"), merge them into a single canonical key. When merging, combine all surface forms from both original keys into one array.
3. Data Integrity (Zero-Loss Policy): Do not drop any surface forms from the input arrays. Do not invent or add new surface forms. Ensure surface forms are unique within their final array (remove duplicates).
4. Semantic Distinction: Do NOT merge words that look similar but have different meanings (e.g., "அண்ணன்" (brother) and "அன்னம்" (swan/food) must remain separate).

### OUTPUT FORMAT
- Return ONLY a valid JSON object. No conversational text, no explanations.
- Structure: {"root_word": ["form1", "form2", ...], ...}

### CRITICAL — NO FORM MAY BE MISSING: Every "form" in the input normalized list MUST appear in exactly one root's array. The union of all value arrays must equal the set of all "form" strings from the input. Do not omit any original form.

"""

# --- Agents: noise_filter → lexical_router → [bridge: morphological_extractor on SPLIT + suffix filter + convert] → root_normalizer → ... ---
noise_filter_agent = Agent(
    name="noise_filter_agent",
    model=model,
    system_prompt=NOISE_FILTER_PROMPT,
    tools=[],
)
lexical_router_agent = Agent(
    name="lexical_router_agent",
    model=model,
    system_prompt=LEXICAL_ROUTER_PROMPT,
    tools=[],
)
morphological_extractor_agent = Agent(
    name="morphological_extractor_agent",
    model=model,
    system_prompt=MORPHOLOGICAL_EXTRACTOR_PROMPT,
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
variant_grouping_validation_agent = Agent(
    name="variant_grouping_validation_agent",
    model=model,
    system_prompt=VARIANT_GROUPING_VALIDATION_PROMPT,
    tools=[],
)

# Two-phase graph: graph1 ends at lexical_router; bridge runs morphological_extractor on SPLIT items, then suffix filter + convert.
builder1 = GraphBuilder()
builder1.add_node(noise_filter_agent, "noise_filter")
builder1.add_node(lexical_router_agent, "lexical_router")
builder1.add_edge("noise_filter", "lexical_router")
builder1.set_entry_point("noise_filter")
graph1 = builder1.build()

builder2 = GraphBuilder()
builder2.add_node(root_normalizer_agent, "root_normalizer")
builder2.add_node(variant_grouping_agent, "variant_grouping")
builder2.add_node(variant_grouping_validation_agent, "variant_grouping_validation")
builder2.add_edge("root_normalizer", "variant_grouping")
builder2.add_edge("variant_grouping", "variant_grouping_validation")
builder2.set_entry_point("root_normalizer")
graph2 = builder2.build()

NODE_ORDER = [
    "noise_filter",
    "lexical_router",
    "morphological_extractor",
    "suffix_filter_bridge",
    "convert_to_split_candidates",
    "root_normalizer",
    "variant_grouping",
    "variant_grouping_validation",
]


def _run_morphological_extractor(split_forms: list[str]) -> list[dict]:
    """Call the Morphological Extractor agent with only SPLIT forms. Returns list of {form, decision, root_words, suffixes}."""
    if not split_forms:
        return []
    user_content = json.dumps({"split_forms": split_forms}, ensure_ascii=False)
    raw = _call_llm_json(MORPHOLOGICAL_EXTRACTOR_PROMPT, user_content)
    parsed = _parse_classifier_output_from_gsr(raw)
    return list(parsed) if parsed else []


def _parse_classifier_output_from_gsr(text: str) -> list[dict] | None:
    """Parse classifier output (from lexical_classifier or similar) into a classifier array (list of {form, decision, root_word|root_words}). Returns None on failure."""
    if not (text or "").strip():
        return None
    raw = text.strip()
    # Strip markdown code fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines)
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Tool result wrapper: {"status": "success", "content": [{"text": "[...]"}]}
            content = data.get("content")
            if isinstance(content, list) and content and isinstance(content[0], dict) and "text" in content[0]:
                inner = json.loads(content[0]["text"])
                if isinstance(inner, list):
                    return inner
            # Other dict: use first list value
            for v in data.values():
                if isinstance(v, list):
                    return v
        return None
    except (json.JSONDecodeError, TypeError):
        return None


def _recover_missing_roots(before_vocab: dict, after_vocab: dict) -> list[tuple[str, list[str]]]:
    """Add roots from before_vocab whose forms are missing in after_vocab. Mutates after_vocab. Returns list of (root, forms) added."""
    forms_after = set()
    for forms in after_vocab.values():
        if isinstance(forms, list):
            forms_after.update(forms)
    added = []
    for root, forms in before_vocab.items():
        if root in after_vocab:
            continue
        if not isinstance(forms, list):
            forms = [forms] if forms else []
        if any(f not in forms_after for f in forms):
            after_vocab[root] = list(dict.fromkeys(forms))
            added.append((root, forms))
    return added


def _bridge_result(text: str):
    """Build a minimal result-like object for bridge steps so _get_agent_output and _format_node_full_output work."""
    msg = {"content": [{"text": text}]}
    result = type("_R", (), {"message": msg, "messages": [msg]})()
    return type("_NodeResult", (), {"result": result})()


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


def _format_node_full_output(node_result) -> str:
    """Format full node output including all agent text, tool calls, and tool results for pipeline.txt."""
    if node_result is None or node_result.result is None:
        return "(no result)"
    parts = []
    # Handle single message or list of messages (multi-turn with tools)
    result = node_result.result
    messages = result.messages if hasattr(result, "messages") and result.messages else ([result.message] if result.message else [])
    for i, msg in enumerate(messages or []):
        if not msg:
            continue
        if i > 0:
            parts.append("\n--- turn %d ---" % (i + 1))
        content = msg.get("content", []) if isinstance(msg, dict) else []
        if not content:
            parts.append(json.dumps(msg, ensure_ascii=False, indent=2))
            continue
        for j, block in enumerate(content):
            if not isinstance(block, dict):
                parts.append("[block %d]: %s" % (j, repr(block)))
                continue
            # Text from agent
            if "text" in block and block["text"]:
                parts.append("[agent output]\n%s" % (block["text"].strip(),))
            # Tool call (tool_use / function_call / name)
            elif block.get("type") == "tool_use" or "tool_use" in block:
                tu = block.get("tool_use", block)
                name = tu.get("name", "?")
                inp = tu.get("input", tu.get("arguments", {}))
                if isinstance(inp, str):
                    try:
                        inp = json.loads(inp)
                    except Exception:
                        pass
                parts.append("[tool call: %s]\n%s" % (name, json.dumps(inp, ensure_ascii=False, indent=2)))
            elif "name" in block and block.get("type") in ("function_call", "tool_use"):
                name = block.get("name", "?")
                inp = block.get("input", block.get("arguments", {}))
                if isinstance(inp, str):
                    try:
                        inp = json.loads(inp)
                    except Exception:
                        pass
                parts.append("[tool call: %s]\n%s" % (name, json.dumps(inp, ensure_ascii=False, indent=2)))
            # Tool result
            elif block.get("type") == "tool_result" or "tool_result" in block:
                tr = block.get("tool_result", block)
                content_list = tr.get("content", [tr.get("output", tr.get("result", ""))])
                if isinstance(content_list, str):
                    content_list = [{"text": content_list}]
                for c in content_list:
                    if isinstance(c, dict) and "text" in c:
                        parts.append("[tool result]\n%s" % (c["text"].strip() or "(empty)"))
                    else:
                        parts.append("[tool result]\n%s" % (json.dumps(tr, ensure_ascii=False, indent=2)))
            elif "content" in block and block.get("type") != "text":
                # Generic content block (e.g. tool result with content array)
                c = block["content"]
                if isinstance(c, list):
                    for item in c:
                        if isinstance(item, dict) and "text" in item and item["text"]:
                            parts.append("[tool result]\n%s" % (item["text"].strip(),))
                else:
                    parts.append("[content]\n%s" % (json.dumps(block, ensure_ascii=False, indent=2)))
            else:
                # Unknown block: show as JSON
                parts.append("[block %d]\n%s" % (j, json.dumps(block, ensure_ascii=False, indent=2)))
    if not parts:
        parts.append(json.dumps(getattr(result, "message", result.__dict__ if hasattr(result, "__dict__") else {}), ensure_ascii=False, indent=2))
    return "\n".join(parts)


if __name__ == "__main__":
    # Run with: python main_candidates_pipeline.py --image path/to/image.png
    #       or: python main_candidates_pipeline.py --pdf path/to/doc.pdf
    # Output: pdf_output/<basename>_page_N_pipeline_output.txt (all agent + tool outputs per node)
    #         pdf_output/<basename>_vocabulary.json (final vocabulary)
    import argparse
    import time
    from pathlib import Path as PathLib

    from word_candidates import ocr_text_to_word_candidates

    parser = argparse.ArgumentParser(
        description="Tamil vocabulary: OCR → word_candidates.py → noise_filter → lexical_router → morphological_extractor (SPLIT only) → [bridge: suffix filter + convert] → root_normalizer → variant_grouping → vocabulary JSON."
    )
    parser.add_argument("--pdf", metavar="PATH", help="Get OCR from PDF (page by page)")
    parser.add_argument("--image", metavar="PATH", help="Get OCR from a single image")
    parser.add_argument(
        "-o", "--output-dir", default="pdf_output", help="Output dir (default: pdf_output)"
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PDF pages (default: 300)")
    parser.add_argument(
        "--page-sleep",
        type=float,
        default=30,
        metavar="SECS",
        help="Seconds to pause after each page to avoid rate limits (default: 30). Set to 0 to disable.",
    )
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

        if " " in args.image and not os.path.isfile(args.image):
            print("Warning: Image path contains a space. If you did not quote it, the path may be wrong.", file=sys.stderr)
            print('  Use quotes, e.g.: --image "test_images/test_image (2).png"', file=sys.stderr)
        if not os.path.isfile(args.image):
            print(f"Warning: Image file not found: {args.image}", file=sys.stderr)
        print("Getting OCR from image (non-agentic)...")
        pages_ocr = [tamil_ocr_from_image(args.image)]
    else:
        parser.error("Input required: use --pdf PATH or --image PATH.")

    input_basename = PathLib(args.pdf).stem if args.pdf else PathLib(args.image).stem
    os.makedirs(args.output_dir, exist_ok=True)

    header = "\n" + "=" * 60 + " PIPELINE OUTPUTS (word_candidates.py + compound-word pipeline) " + "=" * 60
    page_vocabularies = []
    page_vocabularies_before_validation = []
    page_output_paths = []
    graph_failed = False

    for page_num, page_text in enumerate(pages_ocr, start=1):
        print(f"Page {page_num}/{len(pages_ocr)}: word_candidates.py → compound-word pipeline...")
        # 1) Candidates from word_candidates.py (no agents)
        candidates = ocr_text_to_word_candidates(page_text, deduplicate=True)
        if not candidates and (page_text or "").strip():
            print("Warning: OCR returned text but word_candidates produced no candidates. Check word_candidates rules.", file=sys.stderr)
        elif not candidates:
            print("Warning: No word candidates (OCR may be empty or path wrong). Use --image \"path/with spaces.png\" if the path has spaces.", file=sys.stderr)
        initial_input = json.dumps({"candidates": candidates}, ensure_ascii=False)

        try:
            result1 = graph1(initial_input)
        except Exception as e:
            print(f"Error on page {page_num} (graph1): {e}", file=sys.stderr)
            graph_failed = True
            break

        # Bridge: parse lexical_router output → run morphological_extractor on SPLIT items only → merge → suffix filter → convert to split_candidates.
        router_node_result = result1.results.get("lexical_router")
        router_text = _get_agent_output(router_node_result)
        router_output = _parse_classifier_output_from_gsr(router_text)
        if router_output is None and router_text:
            print("Warning: Could not parse lexical_router output as JSON; using empty list.", file=sys.stderr)
        if router_output is None:
            router_output = []
        # Partition KEEP vs SPLIT; run morphological extractor only on SPLIT forms.
        split_forms = [item["form"] for item in router_output if isinstance(item, dict) and item.get("decision") == "SPLIT"]
        extractor_output = _run_morphological_extractor(split_forms) if split_forms else []
        extractor_by_form = {item.get("form"): item for item in extractor_output if isinstance(item, dict) and item.get("form")}
        # Merge: preserve router order; use extractor result for SPLIT items.
        full_classifier_output = []
        for item in router_output:
            if not isinstance(item, dict):
                full_classifier_output.append(item)
                continue
            if item.get("decision") == "KEEP":
                full_classifier_output.append(dict(item))
            else:
                enriched = extractor_by_form.get(item.get("form"))
                if enriched:
                    full_classifier_output.append(enriched)
                else:
                    # Fallback: extractor missed this form; keep as SPLIT with form as single root so it is not dropped
                    full_classifier_output.append({"form": item.get("form", ""), "decision": "SPLIT", "root_words": [item.get("form", "")], "suffixes": []})
        cleaned_classifier = filter_suffixes_from_classifier_output(full_classifier_output)
        split_candidates = classifier_output_to_split_candidates(cleaned_classifier)
        convert_output = json.dumps({"split_candidates": split_candidates}, ensure_ascii=False)
        morphological_extractor_bridge_output = json.dumps(extractor_output, ensure_ascii=False)

        try:
            result2 = graph2(convert_output)
        except Exception as e:
            print(f"Error on page {page_num} (graph2): {e}", file=sys.stderr)
            graph_failed = True
            break

        # Merge results for pipeline output: graph1 nodes, morphological_extractor bridge, suffix_filter_bridge, convert, graph2 nodes.
        merged_results = {}
        for nid in ("noise_filter", "lexical_router"):
            merged_results[nid] = result1.results.get(nid)
        merged_results["morphological_extractor"] = _bridge_result(morphological_extractor_bridge_output)
        merged_results["suffix_filter_bridge"] = _bridge_result(json.dumps(cleaned_classifier, ensure_ascii=False))
        merged_results["convert_to_split_candidates"] = None  # deterministic; no agent result
        for nid in ("root_normalizer", "variant_grouping", "variant_grouping_validation"):
            merged_results[nid] = result2.results.get(nid)

        pipeline_lines = [header]
        # Section: raw OCR (for debugging empty candidates)
        ocr_len = len(page_text or "")
        pipeline_lines.append(f"\n--- OCR (page {page_num}, {ocr_len} chars) ---\n{(page_text or '(empty)')[:5000]}\n")
        if ocr_len > 5000:
            pipeline_lines.append(f"... (truncated; full OCR in {input_basename}_page{page_num}_ocr_raw.txt)\n")
        ocr_raw_path = os.path.join(args.output_dir, f"{input_basename}_page{page_num}_ocr_raw.txt")
        with open(ocr_raw_path, "w", encoding="utf-8") as f:
            f.write(page_text or "")
        # Section: word_candidates (from word_candidates.py)
        word_candidates_block = f"\n--- word_candidates (word_candidates.py) ---\n{initial_input}\n"
        pipeline_lines.append(word_candidates_block)
        print(word_candidates_block)

        variant_grouping_json_text = None
        variant_grouping_before_validation_text = None
        for node_id in NODE_ORDER:
            node_result = merged_results.get(node_id)
            if node_id == "convert_to_split_candidates":
                text = convert_output
                full_out = "[deterministic]\n" + convert_output
            elif node_id == "suffix_filter_bridge":
                text = _get_agent_output(node_result)
                full_out = "[bridge - suffix filter]\n" + (text or "")
            else:
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
                        variant_grouping_before_validation_text = text
                    if node_id == "variant_grouping_validation":
                        variant_grouping_json_text = text
                full_out = _format_node_full_output(node_result)
            # Summary: agent output, or bridge/deterministic output
            output_label = (
                "bridge - suffix filter" if node_id == "suffix_filter_bridge"
                else "bridge - morphological extractor" if node_id == "morphological_extractor"
                else "deterministic" if node_id == "convert_to_split_candidates"
                else "agent output"
            )
            block = f"\n--- {node_id} ---\n[{output_label}]\n{text}\n"
            block += f"\n[full output - agent + tools]\n{full_out}\n"
            print(block)
            pipeline_lines.append(block)

        page_path = os.path.join(
            args.output_dir, f"{input_basename}_page{page_num}_pipeline_output.txt"
        )
        with open(page_path, "w", encoding="utf-8") as f:
            f.write("\n".join(pipeline_lines))
        page_output_paths.append(page_path)
        print(f"Pipeline output saved to: {page_path}")

        if variant_grouping_before_validation_text:
            try:
                data = json.loads(variant_grouping_before_validation_text)
                page_vocabularies_before_validation.append(data)
            except json.JSONDecodeError:
                pass
        if variant_grouping_json_text:
            try:
                data = json.loads(variant_grouping_json_text)
                page_vocabularies.append(data)
            except json.JSONDecodeError:
                pass

        if args.page_sleep > 0 and page_num < len(pages_ocr):
            print(f"Pausing {args.page_sleep}s before next page (rate limit)...", flush=True)
            time.sleep(args.page_sleep)

    # Merge all page vocabularies (before validation)
    before_merged = {}
    for vocab in page_vocabularies_before_validation:
        if not isinstance(vocab, dict):
            continue
        for root, forms in vocab.items():
            if not isinstance(forms, list):
                continue
            before_merged.setdefault(root, []).extend(forms)
    for root in before_merged:
        before_merged[root] = list(dict.fromkeys(before_merged[root]))

    # Merge all page vocabularies (after validation)
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

    recovered = _recover_missing_roots(before_merged, merged)
    if recovered:
        for root, forms in recovered:
            print(f"Recovered root: {root} with forms: {forms}")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"Vocabulary saved to: {output_json_path} ({len(recovered)} root(s) recovered).")
    else:
        print(f"Vocabulary saved to: {output_json_path}")
    print(f"Final root count: {len(merged)}")

    if graph_failed:
        sys.exit(1)
