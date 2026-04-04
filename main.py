import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import requests
from config import MAX_QUESTIONS, RETRY_COUNT , MEDICAL_API_BASE
from state import create_session, get_session, save_answer
from llm import (
    generate_question_raw, generate_diagnosis_raw,
    generate_investigations_raw, generate_medications_raw, generate_procedures_raw,
)
from prompts import (
    build_question_prompt, build_diagnosis_prompt,
    build_investigations_prompt, build_medications_prompt, build_procedures_prompt,
)
from validators import (
    parse_json,
    validate_question, validate_diagnosis,
    validate_investigations, validate_medications, validate_procedures,
)

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s  %(name)s — %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Clinical Triage Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Request models ────────────────────────────────────────────

class StartRequest(BaseModel):
    patient_id:str
    chief_complaint: str
    clinical_history: Optional[str] = ""


class AnswerRequest(BaseModel):
    session_id: str
    selected_option: str

class SelectDiagnosesRequest(BaseModel):
    session_id: str
    selected: List[str]          # doctor's chosen diagnoses

class SelectInvestigationsRequest(BaseModel):
    session_id: str
    selected: List[str]          # doctor's chosen investigations

class SelectMedicationsRequest(BaseModel):
    session_id: str
    selected: List[str]          # doctor's chosen medications


# ── Red-flag detection ────────────────────────────────────────

RED_FLAGS = ["lethargic", "refusing", "seizure", "breathing difficulty",
             "not feeding", "unconscious"]

def _has_red_flag(answers: list) -> bool:
    return any(flag in a.lower() for a in answers for flag in RED_FLAGS)


# ── Generic LLM caller with retries ──────────────────────────

def _generate(raw_fn, prompt_fn, validate_fn, session, label: str) -> dict:
    """
    Call raw_fn(prompt) → parse → validate.
    Retries RETRY_COUNT times on bad output.
    Raises 503 on connection error, 500 if retries exhausted.
    """
    prompt = prompt_fn(session)
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            raw = raw_fn(prompt)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        parsed = parse_json(raw)
        if validate_fn(parsed):
            logger.debug("%s OK on attempt %d", label, attempt)
            return parsed
        logger.warning("%s attempt %d invalid — raw: %s", label, attempt, raw[:200])
    raise HTTPException(status_code=500,
        detail=f"Could not generate valid {label} after {RETRY_COUNT} attempts.")

def _fetch_patient_context(patient_id: str, complaint: str) -> tuple[dict, list]:
    try:
        url = f"{MEDICAL_API_BASE}/api/v1/patient/{patient_id}/complaint/latest"
        print(f"[DEBUG] Calling medical API: {url} with complaint={complaint}")
        response = requests.get(url, params={"complaint": complaint}, timeout=10)
        print(f"[DEBUG] Medical API status code: {response.status_code}")
        if response.status_code == 404:
            print("[DEBUG] No previous consultation found — using empty fallback")
            return {}, []
        response.raise_for_status()
        data = response.json()
        follow_up_history = data.get("follow_up_history", [])
        print(f"[DEBUG] current_consultation visit_date : {data.get('visit_date')}")
        print(f"[DEBUG] current_consultation visit_number: {data.get('visit_number')}")
        print(f"[DEBUG] follow_up_history count       : {len(follow_up_history)}")
        return data, follow_up_history
    except Exception as e:
        print(f"[DEBUG] Exception while fetching patient context: {str(e)}")
        return {}, []
# ── STEP 1 : /start ───────────────────────────────────────────
# Create session, return first triage question.

@app.post("/start")
def start(req: StartRequest):
    print(f"[DEBUG] /start called — patient_id={req.patient_id}, complaint={req.chief_complaint}")
    current_consultation, follow_up_history = _fetch_patient_context(
        req.patient_id, req.chief_complaint
    )
    print(f"[DEBUG] Session will be created with:")
    print(f"[DEBUG]   current_consultation empty : {current_consultation == {}}")
    print(f"[DEBUG]   follow_up_history count : {len(follow_up_history)}")
    sid = create_session(
        req.chief_complaint,
        req.clinical_history or "",
        follow_up_history,
        current_consultation,
    )
    print(f"[DEBUG] Session created — sid={sid}")
    session = get_session(sid)
    print(f"[DEBUG] Session follow_up_history count : {len(session.get('follow_up_history', []))}")
    print(f"[DEBUG] Session current_consultation date  : {session.get('current_consultation', {}).get('visit_date')}")
    q = _generate(generate_question_raw, build_question_prompt, validate_question, session, "question")
    session["pending_question"] = q["question"]
    return {
        "session_id":      sid,
        "question_number": 1,
        "total_questions": MAX_QUESTIONS,
        "question":        q["question"],
        "options":         q["options"],
        "completed":       False,
    }


# ── STEP 2 : /answer (called 5 times) ────────────────────────
# Record answer. After 5 answers → return differential diagnosis list.

@app.post("/answer")
def answer(req: AnswerRequest):
    session = get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    pending_q = session.pop("pending_question", "")
    save_answer(req.session_id, pending_q, req.selected_option)
    red_flag = _has_red_flag(session["answers"])

    # All questions answered → generate diagnosis list
    if session["count"] >= MAX_QUESTIONS:
        diag = _generate(generate_diagnosis_raw, build_diagnosis_prompt,
                         validate_diagnosis, session, "diagnosis")
        return {
            "session_id":     req.session_id,
            "completed":      True,
            "red_flag":       red_flag,
            "considerations": diag["considerations"],
        }

    # Next question (must be unique)
    for attempt in range(RETRY_COUNT):
        q = _generate(generate_question_raw, build_question_prompt,
                      validate_question, session, "question")
        if q["question"] not in session["questions"]:
            session["pending_question"] = q["question"]
            return {
                "session_id":      req.session_id,
                "question_number": session["count"] + 1,
                "total_questions": MAX_QUESTIONS,
                "question":        q["question"],
                "options":         q["options"],
                "completed":       False,
            }
        logger.warning("Duplicate question attempt %d: %s", attempt + 1, q["question"])

    raise HTTPException(status_code=500, detail="Could not generate a unique question.")


# ── STEP 3 : /select-diagnoses ────────────────────────────────
# Doctor selects diagnoses → return suggested investigations.

@app.post("/select-diagnoses")
def select_diagnoses(req: SelectDiagnosesRequest):
    session = get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    if not req.selected:
        raise HTTPException(status_code=400, detail="Select at least one diagnosis.")

    session["diagnoses"] = req.selected

    inv = _generate(generate_investigations_raw, build_investigations_prompt,
                    validate_investigations, session, "investigations")
    return {
        "session_id":     req.session_id,
        "investigations": inv["investigations"],   # [{name, reason}, ...]
    }


# ── STEP 4 : /select-investigations ──────────────────────────
# Doctor selects investigations → return suggested medications.

@app.post("/select-investigations")
def select_investigations(req: SelectInvestigationsRequest):
    session = get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    if not req.selected:
        raise HTTPException(status_code=400, detail="Select at least one investigation.")

    session["investigations"] = req.selected

    meds = _generate(generate_medications_raw, build_medications_prompt,
                     validate_medications, session, "medications")
    return {
        "session_id": req.session_id,
        "medications": meds["medications"],        # [{name, dose, route}, ...]
    }


# ── STEP 5 : /select-medications ─────────────────────────────
# Doctor selects medications → return suggested procedures.

@app.post("/select-medications")
def select_medications(req: SelectMedicationsRequest):
    session = get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    if not req.selected:
        raise HTTPException(status_code=400, detail="Select at least one medication.")

    session["medications"] = req.selected

    proc = _generate(generate_procedures_raw, build_procedures_prompt,
                     validate_procedures, session, "procedures")
    return {
        "session_id": req.session_id,
        "procedures": proc["procedures"],          # [{name, indication}, ...]
    }
# Serve frontend
from fastapi.responses import FileResponse

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")
