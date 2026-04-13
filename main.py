import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import requests
from config import MAX_QUESTIONS, RETRY_COUNT , MEDICAL_API_BASE, QWEN_URL
from state import create_session, get_session, save_answer
from rag_retrival import _clinical_history_chunk
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


@app.get("/health")
def health(request: Request):
    docs_url = str(request.base_url).rstrip("/") + "/docs"
    try:
        response = requests.get(docs_url, timeout=2)
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok" if response.ok else "degraded",
                "service": app.title,
                "checked_url": docs_url,
                "docs_status_code": response.status_code,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "service": app.title,
                "checked_url": docs_url,
                "reason": str(exc),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


def _check_qwen() -> dict:
    response = requests.post(
        QWEN_URL,
        json={
            "prompt": "ping",
            "n_predict": 1,
            "temperature": 0,
            "stop": ["</s>", "<|im_end|>"],
        },
        timeout=5,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "status": "ok",
        "model_url": QWEN_URL,
        "http_status": response.status_code,
        "has_content": "content" in payload,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/qwen-health")
def qwen_health():
    try:
        return JSONResponse(status_code=200, content=_check_qwen())
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "model_url": QWEN_URL,
                "reason": str(exc),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


@app.get("/ready")
def ready(request: Request):
    docs_url = str(request.base_url).rstrip("/") + "/docs"
    try:
        docs_response = requests.get(docs_url, timeout=2)
        qwen_result = _check_qwen()
        status_code = 200 if docs_response.ok and qwen_result["status"] == "ok" else 503
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "ready" if status_code == 200 else "degraded",
                "service": app.title,
                "checked_url": docs_url,
                "docs_status_code": docs_response.status_code,
                "qwen": qwen_result,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "service": app.title,
                "checked_url": docs_url,
                "reason": str(exc),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


# ── Request models ────────────────────────────────────────────

class ManualQA(BaseModel):
    question: str
    answer: str

class Vitals(BaseModel):
    temperatureC: Optional[float] = None
    pulse: Optional[int] = None
    spo2: Optional[int] = None
    systolicBloodPressure: Optional[int] = None
    diastolicBloodPressure: Optional[int] = None
    heightCm: Optional[float] = None
    weightKg: Optional[float] = None
    respiratoryRate: Optional[int] = None
    vitalNotes: Optional[str] = None
    headCircumference: Optional[float] = None
    karnofskyPerformanceScore: Optional[int] = None

class StartRequest(BaseModel):
    patient_id:str
    chief_complaint: str
    complaint_chain: str
    clinical_history: Optional[str] = ""
    manual_key_questions: Optional[List[ManualQA]] = []
    vitals: Optional[Vitals] = None


class AnswerRequest(BaseModel):
    session_id: str
    selected_option: str
    manual_key_questions: Optional[List[ManualQA]] = []

class SelectDiagnosesRequest(BaseModel):
    session_id: str
    selected: List[str]          # doctor's chosen diagnoses
    manual_key_questions: Optional[List[ManualQA]] = []
    manual_diagnoses: Optional[List[str]] = []

class SelectInvestigationsRequest(BaseModel):
    session_id: str
    selected: List[str]          # doctor's chosen investigations
    manual_key_questions: Optional[List[ManualQA]] = []
    manual_investigations: Optional[List[str]] = []

class SelectMedicationsRequest(BaseModel):
    session_id: str
    selected: List[str]          # doctor's chosen medications
    manual_key_questions: Optional[List[ManualQA]] = []
    manual_medications: Optional[List[str]] = []
    manual_procedures: Optional[List[str]] = []


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
    print(f"[DEBUG] SURYA , VIKAS ,SYS PROMPT TESTING FOR GEN Q/A{prompt}") 
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

def _fetch_patient_context(patient_id: str, complaint_chain: str) -> tuple[dict, list]:
    try:
        url = f"{MEDICAL_API_BASE}/api/v1/patient/{patient_id}/complaint/latest"
        print(f"[DEBUG] Calling medical API: {url} with complaint_chain={complaint_chain}")

        response = requests.get(
            url,
            params={"complaint_chain": complaint_chain},
            timeout=10
        )

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
    print(f"[DEBUG] /start called — patient_id={req.patient_id}, complaint={req.complaint_chain}, chief_complaint={req.chief_complaint}")
    current_consultation, follow_up_history = _fetch_patient_context(
    req.patient_id,
    req.complaint_chain
    )

    _,_,patient_history_chunk = _clinical_history_chunk(req.patient_id,req.chief_complaint)

    print(f"[DEBUG] Session will be created with:")
    print(f"[DEBUG] VICKKY")
    print(f"[DEBUG]   current_consultation empty : {current_consultation == {}}")
    print(f"[DEBUG]   follow_up_history count : {len(follow_up_history)}")
    sid = create_session(
        req.chief_complaint,
        patient_history_chunk,
        follow_up_history,
        current_consultation,
        [{"question": mq.question, "answer": mq.answer} for mq in (req.manual_key_questions or [])],
        vitals=req.vitals.dict() if req.vitals else {}
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

    if req.manual_key_questions:
        session["manual_key_questions"] = [
            {"question": mq.question, "answer": mq.answer}
            for mq in req.manual_key_questions
        ]

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
    if req.manual_key_questions:
        session["manual_key_questions"] = [
            {"question": mq.question, "answer": mq.answer}
            for mq in req.manual_key_questions
        ]
    if req.manual_diagnoses:
        session["manual_diagnoses"] = req.manual_diagnoses
    # Merge manual diagnoses into selected for LLM context
    session["diagnoses"] = list(dict.fromkeys(session["diagnoses"] + (req.manual_diagnoses or [])))

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
    if req.manual_key_questions:
        session["manual_key_questions"] = [
            {"question": mq.question, "answer": mq.answer}
            for mq in req.manual_key_questions
        ]
    if req.manual_investigations:
        session["manual_investigations"] = req.manual_investigations
    session["investigations"] = list(dict.fromkeys(session["investigations"] + (req.manual_investigations or [])))

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
    if req.manual_key_questions:
        session["manual_key_questions"] = [
            {"question": mq.question, "answer": mq.answer}
            for mq in req.manual_key_questions
        ]
    if req.manual_medications:
        session["manual_medications"] = req.manual_medications
    if req.manual_procedures:
        session["manual_procedures"] = req.manual_procedures
    session["medications"] = list(dict.fromkeys(session["medications"] + (req.manual_medications or [])))

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
