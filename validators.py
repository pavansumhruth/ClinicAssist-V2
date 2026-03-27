import json
import re


def parse_json(text: str):
    """
    Extract the first complete JSON object from LLM output.
    Handles: markdown fences, preamble text, multiple objects, truncation.
    """
    if not text:
        return None

    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()

    # Find first opening brace
    start = text.find("{")
    if start == -1:
        return None
    text = text[start:]

    # Brace-depth walk to find the first COMPLETE object
    depth, in_str, escape = 0, False, False
    end = None
    for i, ch in enumerate(text):
        if escape:        escape = False; continue
        if ch == "\\" and in_str: escape = True; continue
        if ch == '"':     in_str = not in_str; continue
        if in_str:        continue
        if   ch == "{":   depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        return None  # truncated

    try:
        return json.loads(text[:end])
    except json.JSONDecodeError:
        return None


# ── per-schema validators ─────────────────────────────────────

def validate_question(d) -> bool:
    if not isinstance(d, dict): return False
    q    = d.get("question", "")
    opts = d.get("options", [])
    if not isinstance(q, str) or not q.strip(): return False
    if not isinstance(opts, list): return False
    if not (2 <= len(opts) <= 6): return False
    if not all(isinstance(o, str) and o.strip() for o in opts): return False
    return True


def validate_diagnosis(d) -> bool:
    if not isinstance(d, dict): return False
    items = d.get("considerations", [])
    if not isinstance(items, list) or len(items) < 2: return False
    for item in items:
        if not isinstance(item, dict): return False
        if not item.get("name", "").strip(): return False
        if item.get("likelihood", "").lower() not in {"high", "medium", "low"}: return False
    return True


def validate_investigations(d) -> bool:
    if not isinstance(d, dict): return False
    items = d.get("investigations", [])
    if not isinstance(items, list) or len(items) < 1: return False
    for item in items:
        if not isinstance(item, dict): return False
        if not item.get("name", "").strip(): return False
        if not item.get("reason", "").strip(): return False
    return True


def validate_medications(d) -> bool:
    if not isinstance(d, dict): return False
    items = d.get("medications", [])
    if not isinstance(items, list) or len(items) < 1: return False
    for item in items:
        if not isinstance(item, dict): return False
        if not item.get("name", "").strip(): return False
        if not item.get("dose", "").strip(): return False
        if not item.get("route", "").strip(): return False
    return True


def validate_procedures(d) -> bool:
    if not isinstance(d, dict): return False
    items = d.get("procedures", [])
    if not isinstance(items, list) or len(items) < 1: return False
    for item in items:
        if not isinstance(item, dict): return False
        if not item.get("name", "").strip(): return False
        if not item.get("indication", "").strip(): return False
    return True