import uuid

_sessions: dict = {}


def create_session(complaint: str, history: str,follow_up_history:list,current_consultation:dict) -> str:
    sid = str(uuid.uuid4())
    _sessions[sid] = {
        "complaint":  complaint,
        "history":    history,
        # triage Q&A
        "questions":  [],
        "follow_up_history": follow_up_history or [],
        "current_consultation":  current_consultation or {}, 
        "answers":    [],
        "count":      0,
        # selected values from each step
        "diagnoses":       [],   # list of str  — selected by doctor
        "investigations":  [],   # list of str  — selected by doctor
        "medications":     [],   # list of str  — selected by doctor
        # pending question text (set before /answer is called)
        "pending_question": "",
    }
    return sid


def get_session(sid: str) -> dict | None:
    return _sessions.get(sid)


def save_answer(sid: str, question: str, answer: str):
    s = _sessions[sid]
    s["questions"].append(question)
    s["answers"].append(answer)
    s["count"] += 1