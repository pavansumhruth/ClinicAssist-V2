def _format_follow_up_history(follow_up_history: list) -> str:
    if not follow_up_history:
        return "None"
    parts = []
    for i, entry in enumerate(follow_up_history, 1):
        date       = entry.get("visit_date", "Unknown date")
        visit_no   = entry.get("visit_number", i)
        complaints = ", ".join(entry.get("chief_complaints", []))
        diagnoses  = ", ".join(
            d["name"] for d in entry.get("diagnoses", []) if d.get("selected")
        )
        medications = ", ".join(
            m["name"] for m in entry.get("medications", []) if m.get("selected")
        )
        investigations = ", ".join(
            inv["name"] for inv in entry.get("investigations", []) if inv.get("selected")
        )
        advice     = entry.get("advice", "")
        parts.append(
            f"  Visit {visit_no} ({date}): Complaint: {complaints} | "
            f"Diagnoses: {diagnoses} | Medications: {medications} | "
            f"Investigations: {investigations} | Advice: {advice}"
        )
    return "\n".join(parts)
def _format_current_consultation(last: dict) -> str:
    if not last:
        return "None"
    date         = last.get("visit_date", "Unknown date")
    visit_no     = last.get("visit_number", "")
    complaints   = ", ".join(last.get("chief_complaints", []))
    diagnoses    = ", ".join(
        d["name"] for d in last.get("diagnoses", []) if d.get("selected")
    )
    medications  = ", ".join(
        m["name"] for m in last.get("medications", []) if m.get("selected")
    )
    investigations = ", ".join(
        i["name"] for i in last.get("investigations", []) if i.get("selected")
    )
    key_questions = ", ".join(
        f"{kq['question']} → {kq['answer']}"
        for kq in last.get("key_questions", [])
    )
    advice       = last.get("advice", "")
    vitals       = last.get("vitals") or {}
    vitals_str   = (
        f"Temp: {vitals.get('temp_celsius')}°C, BP: {vitals.get('bp_mmhg')}, "
        f"Weight: {vitals.get('weight_kg')}kg"
        if vitals else "None"
    )
    return (
        f"  Visit {visit_no} ({date}): Complaints: {complaints} | "
        f"Vitals: {vitals_str} | Key Q&A: {key_questions} | "
        f"Diagnoses: {diagnoses} | Investigations: {investigations} | "
        f"Medications: {medications} | Advice: {advice}"
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
You are a clinical triage assistant.
Generate ONE short follow-up question with answer options that helps narrow the diagnosis.

Question format rules:
- Question must be 2 to 6 words ending with "?"
- Examples: "Fever since?", "Peak temperature?", "Activity level?", "Rash present?"

Option format rules:
- Each option is a short label (1 to 5 words)
- Number of options must match the question type naturally:
    Yes / No question      →  2 options  e.g. ["Yes", "No"]
    Duration question      →  4-5 options e.g. ["1 day","2 days","3 days","4+ days"]
    Temperature question   →  5 options  e.g. ["99°F","100°F","101°F","102°F","103°F+"]
    Severity/scale         →  3-4 options e.g. ["Mild","Moderate","Severe"]
    Category question      →  3-5 options e.g. ["None","Localized","Spreading"]
- Do NOT repeat any previously asked question
- If clinical history mentions previous visits, prioritize questions about what has CHANGED since the last visit rather than asking from scratch
<|im_end|>
<|im_start|>user
Chief complaint : {session['complaint']}
Clinical history: {session['history'] or 'None'}
current consultation    : {_format_current_consultation(session.get('current_consultation', {}))}
Follow-up history    : {_format_follow_up_history(session.get('follow_up_history', []))}
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