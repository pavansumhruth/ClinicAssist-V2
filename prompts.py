def _to_names(items) -> str:
    if not isinstance(items, list):
        return "None"
    names = []
    for item in items:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            if name:
                names.append(name)
        elif isinstance(item, str):
            val = item.strip()
            if val:
                names.append(val)
    return ", ".join(names) if names else "None"


def _to_qa(items) -> str:
    if not isinstance(items, list):
        return "None"
    pairs = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if not q and not a:
            continue
        key = (q.lower(), a.lower())
        if key in seen:
            continue
        seen.add(key)
        pairs.append(f"{q} -> {a}")
    return ", ".join(pairs) if pairs else "None"


def _to_complaints(items) -> str:
    if not isinstance(items, list):
        return "None"
    vals = [str(x).strip() for x in items if str(x).strip()]
    return ", ".join(vals) if vals else "None"
def _format_follow_up_history(follow_up_history: list) -> str:
    if not follow_up_history:
        return "None"

    parts = []
    for i, entry in enumerate(follow_up_history, 1):
        if not isinstance(entry, dict):
            continue

        date = entry.get("visit_date", "Unknown date")
        visit_no = entry.get("visit_number", i)

        complaints = _to_complaints(entry.get("chief_complaints", []))
        vitals = entry.get("vitals") or {}
        vitals_str = (
            f"Temp: {vitals.get('temp_celsius')}, BP: {vitals.get('bp_mmhg')}, "
            f"Weight: {vitals.get('weight_kg')}, Height: {vitals.get('height_cm')}, "
            f"HeadCirc: {vitals.get('head_circ_cm')}"
            if isinstance(vitals, dict) and vitals
            else "None"
        )

        key_qa = _to_qa(entry.get("key_questions", []))
        diagnoses = _to_names(entry.get("diagnoses", []))
        investigations = _to_names(entry.get("investigations", []))
        medications = _to_names(entry.get("medications", []))
        procedures = _to_names(entry.get("procedures", []))
        advice = entry.get("advice", "") or "None"
        follow_up_date = entry.get("follow_up_date", "") or "None"

        parts.append(
            f"Visit {visit_no} ({date}) | Complaints: {complaints} | "
            f"Vitals: {vitals_str} | Key Q&A: {key_qa} | "
            f"Diagnoses: {diagnoses} | Investigations: {investigations} | "
            f"Medications: {medications} | Procedures: {procedures} | "
            f"Advice: {advice} | Follow-up date: {follow_up_date}"
        )

    return "\n".join(parts) if parts else "None"
def _format_current_consultation(last: dict) -> str:
    if not isinstance(last, dict) or not last:
        return "None"

    date = last.get("visit_date", "Unknown date")
    visit_no = last.get("visit_number", "")

    complaints = _to_complaints(last.get("chief_complaints", []))
    diagnoses = _to_names(last.get("diagnoses", []))
    investigations = _to_names(last.get("investigations", []))
    medications = _to_names(last.get("medications", []))
    procedures = _to_names(last.get("procedures", []))
    key_qa = _to_qa(last.get("key_questions", []))

    advice = last.get("advice", "") or "None"
    follow_up_date = last.get("follow_up_date", "") or "None"

    vitals = last.get("vitals") or {}
    vitals_str = (
        f"Temp: {vitals.get('temp_celsius')}, BP: {vitals.get('bp_mmhg')}, "
        f"Weight: {vitals.get('weight_kg')}, Height: {vitals.get('height_cm')}, "
        f"HeadCirc: {vitals.get('head_circ_cm')}"
        if isinstance(vitals, dict) and vitals
        else "None"
    )

    return (
        f"Visit {visit_no} ({date}) | Complaints: {complaints} | "
        f"Vitals: {vitals_str} | Key Q&A: {key_qa} | "
        f"Diagnoses: {diagnoses} | Investigations: {investigations} | "
        f"Medications: {medications} | Procedures: {procedures} | "
        f"Advice: {advice} | Follow-up date: {follow_up_date}"
    )
def build_question_prompt(session: dict) -> str:
    """
    Build the prompt for the next triage question.
    The GBNF grammar (not the prompt) enforces JSON structure,
    so this prompt focuses purely on clinical context.
    """
    prev = ""
    if session["questions"]:
        pairs = "\n".join(
            f"  Q: {q}\n  A: {a}"
            for q, a in zip(session["questions"], session["answers"])
        )
        prev = f"Questions already asked:\n{pairs}"
    else:
        prev = "Questions already asked: none"

    return f"""<|im_start|>system
You are a clinical triage assistant helping a doctor refine diagnosis.

YOUR GOAL:
Ask ONE highly relevant follow-up question that provides NEW clinical information.

━━━━━━━━━━━━━━━━━━━
CRITICAL RULES
━━━━━━━━━━━━━━━━━━━

LANGUAGE:
- Respond ONLY in English
- Use simple medical English
- Do NOT use any non-English characters

CLINICAL INTELLIGENCE:
- Carefully read ALL inputs before asking a question
- Extract known facts internally (do NOT output them)
- Identify what information is already known

STRICT PROHIBITIONS:
- DO NOT ask questions if the answer is already known
- DO NOT repeat or rephrase previously asked questions
- DO NOT ask obvious or redundant questions
- DO NOT ask about information already present in:
  • Chief complaint
  • Clinical history
  • Current consultation
  • Follow-up history

QUESTION QUALITY:
- Ask only about MISSING or UNKNOWN information
- Question must help narrow diagnosis
- Prefer high-value clinical discriminators:
  • severity 
  • associated symptoms
  • progression
  • risk factors
  • red flags

━━━━━━━━━━━━━━━━━━━
QUESTION FORMAT
━━━━━━━━━━━━━━━━━━━
- 2 to 6 words
- Must end with "?"
- Clear, specific, and clinically meaningful

GOOD EXAMPLES:
- "Peak temperature?"
- "Chills present?"
- "Rash location?"
- "Vomiting present?"
- "Recent travel?"

BAD EXAMPLES:
- "Fever since?" (if duration already known)
- "Pain present?" (too obvious)
- "How are you feeling?" (not clinical)

━━━━━━━━━━━━━━━━━━━
OPTIONS FORMAT
━━━━━━━━━━━━━━━━━━━
- Each option: 1 to 5 words
- Match question type naturally:

Yes/No → ["Yes", "No"]

Duration → ["1 day","2 days","3 days","4+ days"]

Temperature → ["99°F","100°F","101°F","102°F","103°F+"]

Severity → ["Mild","Moderate","Severe"]

Category → ["None","Localized","Spreading"]

━━━━━━━━━━━━━━━━━━━
DECISION LOGIC (IMPORTANT)
━━━━━━━━━━━━━━━━━━━
Before generating a question:
1. Identify known facts
2. Identify missing information
3. Choose the MOST useful missing variable
4. Ask about that ONLY

If fever duration is already known:
→ DO NOT ask duration again
→ Ask about severity or associated symptoms instead

━━━━━━━━━━━━━━━━━━━
OUTPUT
━━━━━━━━━━━━━━━━━━━
Return ONLY the question and options.
No explanation.
No extra text.

<|im_end|>
<|im_start|>user
Chief complaint : {session['complaint']} 
Clinical history: {session['history'] or 'None'}
Current consultation:
{_format_current_consultation(session.get('current_consultation', {}))}
Follow-up history:
{_format_follow_up_history(session.get('follow_up_history', []))}

Questions already asked:
{prev}
<|im_end|>
<|im_start|>assistant
"""

def build_diagnosis_prompt(session: dict) -> str:
    qa = "\n".join(f"  {q} → {a}" for q, a in zip(session["questions"], session["answers"]))
    return f"""<|im_start|>system
You are a clinical assistant. List 6-10 differential diagnoses ranked most to least likely.
Use only "high", "medium", or "low" for likelihood. Common conditions first.
<|im_end|>
<|im_start|>user
Chief complaint : {session['complaint']}
Clinical history: {session['history'] or 'None'}
current consultation    : {_format_current_consultation(session.get('current_consultation', {}))}
Follow-up history    : {_format_follow_up_history(session.get('follow_up_history', []))}
Clinical Q&A:
{qa}
<|im_end|>
<|im_start|>assistant
"""


def build_investigations_prompt(session: dict) -> str:
    diagnoses = ", ".join(session["diagnoses"])
    qa = "\n".join(f"  {q} → {a}" for q, a in zip(session["questions"], session["answers"]))
    return f"""<|im_start|>system
You are a clinical assistant. Suggest 5-10 relevant medical investigations for the given diagnoses.
For each investigation provide a short reason (why it is needed). Be specific and practical.
<|im_end|>
<|im_start|>user
Chief complaint : {session['complaint']}
Clinical history: {session['history'] or 'None'}
current consultation    : {_format_current_consultation(session.get('current_consultation', {}))}
Follow-up history    : {_format_follow_up_history(session.get('follow_up_history', []))}
Clinical Q&A:
{qa}
Selected diagnoses: {diagnoses}
<|im_end|>
<|im_start|>assistant
"""


def build_medications_prompt(session: dict) -> str:
    diagnoses      = ", ".join(session["diagnoses"])
    investigations = ", ".join(session["investigations"])
    return f"""<|im_start|>system
You are a clinical assistant. Suggest 5-10 medications appropriate for the diagnoses and investigation findings.
For each medication provide the typical adult dose and route of administration.
<|im_end|>
<|im_start|>user
Chief complaint    : {session['complaint']}
Clinical history   : {session['history'] or 'None'}
current consultation    : {_format_current_consultation(session.get('current_consultation', {}))}
Follow-up history    : {_format_follow_up_history(session.get('follow_up_history', []))}
Diagnoses          : {diagnoses}
Investigations done: {investigations}
<|im_end|>
<|im_start|>assistant
"""


def build_procedures_prompt(session: dict) -> str:
    diagnoses      = ", ".join(session["diagnoses"])
    investigations = ", ".join(session["investigations"])
    medications    = ", ".join(session["medications"])
    return f"""<|im_start|>system
You are a clinical assistant. Suggest 3-8 clinical procedures or interventions indicated for this patient.
For each procedure give a clear indication (why it is needed for this patient).
<|im_end|>
<|im_start|>user
Chief complaint    : {session['complaint']}
Clinical history   : {session['history'] or 'None'}
current consultation    : {_format_current_consultation(session.get('current_consultation', {}))}
Follow-up history    : {_format_follow_up_history(session.get('follow_up_history', []))}
Diagnoses          : {diagnoses}
Investigations done: {investigations}
Medications given  : {medications}
<|im_end|>
<|im_start|>assistant
"""
