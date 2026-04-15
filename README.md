# ClinicAssist V2 — Clinical Triage Assistant

> AI-assisted differential diagnosis and clinical management support for clinicians.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack](#2-tech-stack)
3. [Project Folder Structure](#3-project-folder-structure)
4. [Installation and Setup](#4-installation-and-setup)
5. [Environment Variables](#5-environment-variables)
6. [High Level Design](#6-high-level-design)
7. [Low Level Design](#7-low-level-design)
8. [API Endpoints](#8-api-endpoints)
9. [Step-by-Step Flow](#9-step-by-step-flow)
10. [Prompt Design](#10-prompt-design)
11. [Error Handling and Fallbacks](#11-error-handling-and-fallbacks)
12. [Known Limitations and Roadmap](#12-known-limitations-and-roadmap)
13. [Changelog](#13-changelog)

---

## 1. Project Overview

**ClinicAssist V2** is a clinician-facing, AI-powered triage and clinical decision support system. It is designed for use at the point of care — in clinics, hospitals, or telemedicine settings — to help doctors work through a structured clinical consultation faster and more consistently.

The system guides the clinician through a 5-question adaptive triage interview, then generates ranked differential diagnoses, suggested investigations, appropriate medications (with dose and route), and recommended clinical procedures — all driven by a locally hosted large language model (LLM) running through **llama.cpp**.

**Who it is for**: Medical practitioners (doctors, clinical officers) who want AI-assisted decision support embedded directly into their workflow.

**What problem it solves**: In busy clinical settings, recalling the full breadth of differential diagnoses and matching investigation/medication choices is cognitively demanding. ClinicAssist V2 acts as an intelligent co-pilot, enriching each LLM query with the patient's past consultation history and current visit data pulled from an internal medical backend, so that suggestions are always contextualised to the specific patient — not just to the presenting complaint in isolation.

---

## 2. Tech Stack

| Layer | Technology |
|---|---|
| **Backend API Framework** | [FastAPI](https://fastapi.tiangolo.com/) |
| **ASGI Server** | [Uvicorn](https://www.uvicorn.org/) |
| **Request Validation** | [Pydantic v2](https://docs.pydantic.dev/) (via FastAPI) |
| **HTTP Client** | [requests](https://docs.python-requests.org/) |
| **LLM Runtime** | [llama.cpp](https://github.com/ggerganov/llama.cpp) HTTP server (`/completion` API) |
| **LLM Model** | Qwen (served at `QWEN_URL`, identified by the llama.cpp endpoint) |
| **Constrained LLM Decoding** | GBNF grammars (Backus-Naur Form for JSON schema enforcement) |
| **Session Storage** | In-process Python dict (ephemeral, no external database) |
| **Frontend** | Vanilla HTML5 + CSS3 + Vanilla JavaScript (single `index.html`) |
| **Frontend Fonts** | [IBM Plex Sans](https://fonts.google.com/specimen/IBM+Plex+Sans) + [IBM Plex Mono](https://fonts.google.com/specimen/IBM+Plex+Mono) via Google Fonts |
| **Medical Data Backend** | Internal REST API at `MEDICAL_API_BASE` (separate service, `http://34.180.37.249:8007`) |
| **AI Notes Streaming** | Separate optional autocomplete service at `BASE_AUTOCOMPLETE` (`http://34.180.37.249:8004`) via SSE (Server-Sent Events). In this deployment, port 8004 may be unavailable. |
| **Python** | 3.12 (derived from `auto_comp/pyvenv.cfg`) |
| **CORS Middleware** | FastAPI `CORSMiddleware` (all origins allowed) |
| **Logging** | Python `logging` (DEBUG level) |

**Python packages** (derived from `Read_Me` and `auto_comp/bin` contents — no `requirements.txt` exists):

- `fastapi`
- `uvicorn`
- `requests`
- `pydantic` (bundled with FastAPI)

---

## 3. Project Folder Structure

```
ClinicAssist-V2/
│
├── main.py                  # FastAPI application entrypoint. Defines all 6 endpoints,
│                            # the red-flag detector, the generic LLM caller with retries,
│                            # and the medical backend fetch helper.
│
├── config.py                # All configuration constants: LLM URL, Medical API base URL,
│                            # per-task token limits, timeouts, temperature, retry count,
│                            # and MAX_QUESTIONS session limit. No env vars — all hardcoded.
│
├── state.py                 # In-memory session store. Holds the _sessions dict, and
│                            # exposes create_session(), get_session(), save_answer().
│                            # Sessions are lost on server restart.
│
├── prompts.py               # Five prompt-builder functions (one per LLM task) plus two
│                            # private formatters for follow-up history and current
│                            # consultation data. Uses ChatML format (<|im_start|> tags).
│
├── llm.py                   # Thin HTTP client. One private _call() fn that posts to the
│                            # llama.cpp /completion endpoint with grammar + token limit +
│                            # timeout. Five public wrappers (one per task) with their
│                            # specific limits.
│
├── grammar.py               # GBNF grammar strings for constrained decoding. One grammar
│                            # per output type: QUESTION, DIAGNOSIS, INVESTIGATIONS,
│                            # MEDICATIONS, PROCEDURES. Enforces exact JSON schema at
│                            # the token level — the model cannot produce invalid JSON.
│
├── validators.py            # JSON parsing and per-schema validation. parse_json() extracts
│                            # the first complete JSON object from raw LLM output using a
│                            # brace-depth walk (handles preamble, markdown fences,
│                            # truncation). Five validate_*() functions check field
│                            # presence, types, and constraints.
│
├── index.html               # Complete single-file frontend. 1383 lines of HTML + CSS +
│                            # vanilla JS. Implements a 7-step clinical workflow: Input →
│                            # Questions → Diagnosis → Investigations → Medications →
│                            # Procedures → Summary. Connects to both the triage API
│                            # (port 9000) and an optional autocomplete/notes API (port 8004).
│                            # Includes a hidden debug console panel.
│
├── Benchmark.py             # Standalone benchmarking script. Runs 3 test calls per
│                            # category (question with grammar, question without grammar,
│                            # diagnosis, investigations) against the Qwen completion URL
│                            # from config.py (currently port 8006) and
│                            # recommends timeout values to paste into config.py.
│
├── Read_Me                  # Original plain-text quick-start notes (not a proper README).
│                            # Contains the basic commands to start uvicorn, activate the
│                            # venv, and start llama.cpp.
│
├── .gitignore               # Excludes __pycache__/ and .idea/ from version control.
│
├── main.py.backup           # Backup of main.py before patient_id and medical API
│                            # integration were added. The old version had no MEDICAL_API_BASE
│                            # import and no _fetch_patient_context() call.
│
├── prompts.py.backup        # Backup of prompts.py before follow-up history and current
│                            # consultation context were injected. The old prompts only
│                            # used complaint, history, and Q&A pairs.
│
├── index.html.backup        # Backup of index.html (functionally near-identical to
│                            # current but lacks some UI refinements).
│
└── auto_comp/               # Python virtual environment directory (Python 3.12).
    ├── bin/                 # Activate scripts (bash/fish/csh/PowerShell), pip binaries.
    ├── lib/                 # Installed site-packages (fastapi, uvicorn, etc.).
    └── pyvenv.cfg           # Venv config: Python 3.12, system packages not included.
```

---

## 4. Installation and Setup

### Prerequisites

- Python 3.12+
- `llama.cpp` compiled and running with a Qwen-compatible GGUF model
- Access to the internal medical backend API (currently hardcoded to `34.180.37.249:8007`)

### Step 1 — Clone the repository

```bash
git clone <your-repo-url>
cd ClinicAssist-V2
```

### Step 2 — Create and activate a virtual environment

```bash
python3 -m venv auto_comp
source auto_comp/bin/activate
```

### Step 3 — Install Python dependencies

There is no `requirements.txt` in the project. Install manually:

```bash
pip install fastapi uvicorn requests
```

### Step 4 — Start the llama.cpp server (Qwen model)

Run llama.cpp's built-in HTTP server on port 8006 (or whatever matches `QWEN_URL` in `config.py`):

```bash
cd ~/llama.cpp/build/bin
./llama-server -m ~/models/qwen.gguf -c 4096 -t 8 --parallel 1 --port 8006
```

> The model file and path are not defined in this repository. You must supply your own Qwen GGUF model. The original `Read_Me` references a `BioMistral-7B.Q4_K_M.gguf` at port 8081 — that is from an older version. The current `config.py` points to port 8006 on `34.180.37.249`.

### Step 5 — (Optional) Tune timeouts with Benchmark.py

Before your first run, calibrate timeouts against your hardware:

```bash
python3 Benchmark.py
```

Paste the recommended values into `config.py`.

### Step 6 — Start the FastAPI backend

```bash
uvicorn main:app --reload --log-level debug --port 9000
```

> The frontend in `index.html` is hardcoded to call `http://34.180.37.249:9000` as `BASE_TRIAGE`. If you run locally, you must either update that constant in `index.html` or expose the server on that IP.

### Step 7 — Serve the frontend

The FastAPI app serves `index.html` directly at the root path `/` via `FileResponse`. Simply open:

```
http://34.180.37.249:9000/
```

Or serve it separately (for development):

```bash
python3 -m http.server 5500
# then open http://34.180.37.249:5500
```

---

## 5. Environment Variables

**There are no environment variables in this project.** All configuration is hardcoded directly in `config.py`. There is no `.env` file and no `python-dotenv` usage.

The following constants in `config.py` should be treated as the effective configuration surface. To change them, edit the file directly:

| Constant | Default Value | Description |
|---|---|---|
| `QWEN_URL` | `http://34.180.37.249:8006/completion` | The llama.cpp HTTP server `/completion` endpoint for the Qwen model. |
| `MEDICAL_API_BASE` | `http://34.180.37.249:8007` | Base URL of the internal medical data backend that stores patient consultation history. |
| `MAX_TOKENS_QUESTION` | `128` | Maximum tokens the LLM can generate for a triage question. |
| `MAX_TOKENS_DIAGNOSIS` | `350` | Maximum tokens for a differential diagnosis list. |
| `MAX_TOKENS_INVESTIGATIONS` | `400` | Maximum tokens for an investigations list. |
| `MAX_TOKENS_MEDICATIONS` | `400` | Maximum tokens for a medications list. |
| `MAX_TOKENS_PROCEDURES` | `350` | Maximum tokens for a procedures list. |
| `TEMPERATURE` | `0.3` | LLM sampling temperature applied to all calls. |
| `RETRY_COUNT` | `5` | Number of attempts the `_generate()` helper makes before raising HTTP 500. |
| `TIMEOUT_QUESTION` | `30` | HTTP timeout (seconds) for question generation calls. |
| `TIMEOUT_DIAGNOSIS` | `60` | HTTP timeout for diagnosis generation. |
| `TIMEOUT_INVESTIGATIONS` | `120` | HTTP timeout for investigations generation. |
| `TIMEOUT_MEDICATIONS` | `120` | HTTP timeout for medications generation. |
| `TIMEOUT_PROCEDURES` | `60` | HTTP timeout for procedures generation. |
| `MAX_QUESTIONS` | `5` | Number of triage questions asked before triggering diagnosis generation. |

**Frontend constants** (hardcoded in `index.html`, not in Python config):

| JS Constant | Value | Description |
|---|---|---|
| `BASE_TRIAGE` | `http://34.180.37.249:9000` | The ClinicAssist FastAPI backend base URL. |
| `BASE_AUTOCOMPLETE` | `http://34.180.37.249:8004` | Separate optional AI notes/autocomplete streaming service (currently unreachable if port 8004 is down). |
| `PATIENT_ID` | `791427b4-9cc4-8bcc-3fee-e3e14b6d3fea` | Hardcoded patient UUID sent with every `/start` call and every AI Notes generation call. |
| `MAX_Q` | `5` | Mirrors `MAX_QUESTIONS` on the frontend. |

---

## 6. High Level Design

### Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                         BROWSER (Clinician)                         │
│                                                                     │
│   index.html (Vanilla JS)                                           │
│   ┌─────────────┐    ┌──────────────────────────────────────────┐  │
│   │ Triage Flow │    │  AI Notes (SSE stream, section-by-section)│  │
│   └──────┬──────┘    └────────────────┬─────────────────────────┘  │
└──────────┼──────────────────────────── ┼───────────────────────────┘
           │  REST (JSON)                │  SSE Stream
           ▼                             ▼
┌──────────────────────┐     ┌────────────────────────────┐
│  ClinicAssist V2     │     │  Autocomplete/Notes Service │
│  FastAPI (port 9000) │     │  (port 8004) — SEPARATE    │
│                      │     │  Optional and may be down  │
│                      │     │  Not part of this repo      │
│  main.py             │     └────────────────────────────┘
│  ├─ /start           │
│  ├─ /answer          │  calls
│  ├─ /select-diagnoses│──────────────────────────────────────────────►
│  ├─ /select-         │                               ┌──────────────┐
│  │   investigations  │                               │ Medical API  │
│  ├─ /select-         │  GET /api/v1/patient/         │ (port 8007)  │
│  │   medications     │  {id}/complaint/latest        │              │
│  └─ GET /            │◄──────────────────────────────│  Patient     │
│                      │                               │  history,    │
│  Calls LLM via       │                               │  follow-up   │
│  llm.py + grammar.py │                               └──────────────┘
└──────────┬───────────┘
           │  HTTP POST /completion (with GBNF grammar)
           ▼
┌────────────────────────┐
│  llama.cpp HTTP Server │
│  Qwen GGUF model       │
│  (port 8006)           │
└────────────────────────┘
```

### Request Flow (Triage Path)

1. **Browser → POST `/start`**: The clinician enters a chief complaint and optional history. The frontend sends `patient_id`, `chief_complaint`, and `clinical_history`.
2. **Backend → Medical API**: ClinicAssist fetches the patient's latest consultation and full follow-up history from the internal medical backend.
3. **Backend → Session creation**: A new in-memory session is created with all patient context stored.
4. **Backend → LLM**: The question prompt (with patient history injected) is sent to llama.cpp. The GBNF grammar ensures the response is always valid JSON.
5. **Backend → Browser**: First triage question + options returned.
6. **Browser → POST `/answer` × 5**: Clinician selects an option per question. After 5 answers, the backend triggers diagnosis generation.
7. **Browser → POST `/select-diagnoses`** → investigations generated by LLM.
8. **Browser → POST `/select-investigations`** → medications generated by LLM.
9. **Browser → POST `/select-medications`** → procedures generated by LLM.
10. **Browser renders Summary**: The clinician sees the final management plan. They can print or restart.

---

## 7. Low Level Design

### 7.1 Session Creation and Management

Sessions are managed entirely in `state.py` using a module-level dict `_sessions: dict = {}`.

**`create_session(complaint, history, follow_up_history, current_consultation) → str`**
- Generates a UUID4 as the session ID.
- Stores a dict with:
  - `complaint` — the chief complaint string
  - `history` — optional clinical history string (empty string if not provided)
  - `questions` — empty list, grows as Q&A pairs accumulate
  - `answers` — parallel list to `questions`
  - `count` — integer, incremented by `save_answer()` after each answer
  - `follow_up_history` — list of past visit dicts from the medical API (may be empty)
  - `current_consultation` — dict of the most recent visit from the medical API (may be empty)
  - `diagnoses`, `investigations`, `medications` — empty lists, populated by doctor selections
  - `pending_question` — the last question text sent to the frontend, used to pair with the incoming answer

**`get_session(sid) → dict | None`**
- Plain dict lookup from `_sessions`. Returns `None` if not found.

**`save_answer(sid, question, answer)`**
- Appends `question` and `answer` to the respective lists.
- Increments `count`.

> **Important**: Sessions are stored in-process memory only. They are lost on server restart and are not shared across worker processes. There is no TTL or expiry mechanism.

---

### 7.2 LLM Calls and Retry Logic

The generic retry wrapper is `_generate()` in `main.py`:

```
_generate(raw_fn, prompt_fn, validate_fn, session, label)
  └─ prompt = prompt_fn(session)
  └─ for attempt in range(1, RETRY_COUNT + 1):
       └─ raw = raw_fn(prompt)          # calls LLM, may raise RuntimeError
       └─ parsed = parse_json(raw)      # brace-depth JSON extractor
       └─ if validate_fn(parsed): return parsed
       └─ log warning + retry
  └─ raise HTTP 500 if all retries exhausted
```

**The same prompt is reused across all retries** — the retry is not prompt-adaptive. If the LLM repeatedly returns bad JSON, the retries will all fail identically (unless the LLM introduces natural token-level variation at temperature 0.3).

`llm.py._call()` does the actual HTTP POST to llama.cpp:
- Payload: `{prompt, n_predict, temperature, stop, grammar}`
- Stop tokens: `["</s>", "<|im_end|>"]` — the ChatML end-of-turn marker.
- The `grammar` field activates llama.cpp's constrained decoding. The model can only emit tokens that are valid within the GBNF grammar, so malformed JSON is structurally impossible — but the retry logic still exists as a safety net for edge cases.

**Error mapping in `_call()`**:
- `ConnectionError` → `RuntimeError("Cannot reach Qwen…")`
- `Timeout` → `RuntimeError("Qwen timed out…")`
- `HTTPError` → `RuntimeError("Qwen HTTP error: …")`
- `KeyError/ValueError` on `r.json()["content"]` → `RuntimeError("Unexpected response…")`

`_generate()` then converts any `RuntimeError` from `raw_fn()` into `HTTP 503`.

---

### 7.3 Prompt Construction

All prompts use ChatML format: `<|im_start|>system … <|im_end|>` and `<|im_start|>user … <|im_end|>`, and the assistant turn is left open (`<|im_start|>assistant\n`) so llama.cpp continues from it.

Each prompt builder in `prompts.py` receives the full session dict and assembles context from it. Two private helpers format the patient history:

**`_format_follow_up_history(follow_up_history: list) → str`**
- Iterates the list of past visit dicts.
- For each visit, extracts: `visit_date`, `visit_number`, `chief_complaints[]`, selected `diagnoses[].name`, selected `medications[].name`, selected `investigations[].name`, and `advice`.
- Returns a multi-line string, one line per visit. Returns `"None"` if empty.

**`_format_current_consultation(last: dict) → str`**
- Extracts from the most recent consultation: `visit_date`, `visit_number`, `chief_complaints[]`, selected `diagnoses[].name`, selected `medications[].name`, selected `investigations[].name`, `key_questions[]` (formatted as `question → answer`), `vitals` (temp, BP, weight), and `advice`.
- Returns a single summary line. Returns `"None"` if empty.

Both helpers filter selections by `item.get("selected")` being truthy — only items the doctor actually confirmed in the last visit are included.

---

### 7.4 Medical Backend API Call

`_fetch_patient_context(patient_id, complaint)` in `main.py`:

- Makes a `GET` request to:
  ```
  {MEDICAL_API_BASE}/api/v1/patient/{patient_id}/complaint/latest
  ```
  with query parameter `?complaint={complaint}` and a 10-second timeout.
- **404 response**: treated as "no previous consultation found" — returns `({}, [])` (empty dict, empty list). This is the expected case for new patients.
- **Other error or exception**: also returns `({}, [])` silently (bare `except Exception`). This means a medical API outage degrades gracefully — the session proceeds without historical context rather than failing.
- **Success**: returns `(data, data.get("follow_up_history", []))` where `data` is the full response JSON (the current consultation) and the follow-up history is a list of previous visit dicts embedded within it.

---

### 7.5 Follow-up History and Consultation Injection into Prompts

After `_fetch_patient_context()` returns, `main.py:/start` calls `create_session()` passing both `current_consultation` (the current visit dict) and `follow_up_history` (the list of prior visits). These are stored on the session dict.

Every one of the five prompt builders reads from the session:
```python
session.get('current_consultation', {})
session.get('follow_up_history', [])
```
and injects them into the `<|im_start|>user` section as:
```
current consultation    : <formatted one-liner>
Follow-up history    : <formatted multi-line or "None">
```

This means the LLM sees the full patient clinical narrative — not just the current complaint — when generating questions, diagnoses, investigations, medications, and procedures.

---

### 7.6 Validation

Each LLM response type has a dedicated validator in `validators.py`.

**`parse_json(text) → dict | None`**
- Strips markdown code fences (` ```json `, ` ``` `).
- Finds the first `{` character.
- Walks character-by-character tracking brace depth, string state, and escape state to find the closing `}` of the first complete JSON object.
- Returns `json.loads()` of that substring, or `None` on failure.
- Handles: preamble text before JSON, trailing text after the object, incomplete/truncated output, and nested objects.

**`validate_question(d) → bool`**: Requires `question` (non-empty string) and `options` (list of 2–6 non-empty strings).

**`validate_diagnosis(d) → bool`**: Requires `considerations` (list of ≥2 items), each with a non-empty `name` and `likelihood` in `{"high", "medium", "low"}`.

**`validate_investigations(d) → bool`**: Requires `investigations` (list of ≥1 items), each with non-empty `name` and `reason`.

**`validate_medications(d) → bool`**: Requires `medications` (list of ≥1 items), each with non-empty `name`, `dose`, and `route`.

**`validate_procedures(d) → bool`**: Requires `procedures` (list of ≥1 items), each with non-empty `name` and `indication`.

---

### 7.7 Red Flag Detection

`_has_red_flag(answers: list) → bool` in `main.py`:

The hardcoded red-flag keyword list is:
```python
RED_FLAGS = ["lethargic", "refusing", "seizure", "breathing difficulty",
             "not feeding", "unconscious"]
```

It checks whether any selected answer (lowercased) contains any of these substrings. This is a **simple substring match** — no NLP, no medical ontology. It is called after each answer is recorded in `/answer`, and the result is included in the final diagnosis response as `red_flag: bool`. The frontend shows a red banner (`⚠️ Red-flag symptom detected`) if this is `true`.

---

## 8. API Endpoints

All endpoints accept and return `application/json`. CORS is open to all origins.

---

### `GET /`
**Description**: Serves the `index.html` frontend file.
**Response**: HTML file (`text/html`).

---

### `POST /start`

**Description**: Begins a new triage session. Fetches the patient's latest consultation and follow-up history from the medical backend, creates an in-memory session, and returns the first triage question.

**Request Body** (`StartRequest`):

| Field | Type | Required | Description |
|---|---|---|---|
| `patient_id` | `str` | ✅ | The patient's UUID used to query the medical backend. |
| `chief_complaint` | `str` | ✅ | The presenting complaint (e.g. "fever with cough"). |
| `clinical_history` | `str` | ❌ (default `""`) | Optional free-text clinical history (age, comorbidities, allergies, etc.). |

**Response Body**:

| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | UUID of the newly created session. Needed for all subsequent calls. |
| `question_number` | `int` | Always `1`. |
| `total_questions` | `int` | Always `5` (value of `MAX_QUESTIONS`). |
| `question` | `str` | The first triage question (2–6 words, ends with `?`). |
| `options` | `list[str]` | 2–6 short answer options. |
| `completed` | `bool` | Always `false` here. |

**Errors**:
- `503` if the LLM is unreachable.
- `500` if all retries for question generation fail.

---

### `POST /answer`

**Description**: Records the clinician's answer to the current question. If fewer than `MAX_QUESTIONS` answers have been given, returns the next question. After the 5th answer, generates the differential diagnosis list.

**Request Body** (`AnswerRequest`):

| Field | Type | Required | Description |
|---|---|---|---|
| `session_id` | `str` | ✅ | The session UUID from `/start`. |
| `selected_option` | `str` | ✅ | The exact text of the selected option. |

**Response Body (question not yet complete)**:

| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | Echo of input session ID. |
| `question_number` | `int` | The number of the next question (2–5). |
| `total_questions` | `int` | Always `5`. |
| `question` | `str` | Next question text. |
| `options` | `list[str]` | Answer options. |
| `completed` | `bool` | `false`. |

**Response Body (after 5th answer)**:

| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | Session ID. |
| `completed` | `bool` | `true`. |
| `red_flag` | `bool` | Whether any red-flag keyword was found in the collected answers. |
| `considerations` | `list[{name: str, likelihood: str}]` | Ranked differential diagnoses. `likelihood` is `"high"`, `"medium"`, or `"low"`. |

**Errors**:
- `404` — session not found.
- `500` — could not generate a unique question after `RETRY_COUNT` attempts.
- `503` — LLM unreachable.

---

### `POST /select-diagnoses`

**Description**: Records the doctor's chosen diagnoses and returns suggested investigations.

**Request Body** (`SelectDiagnosesRequest`):

| Field | Type | Required | Description |
|---|---|---|---|
| `session_id` | `str` | ✅ | Session UUID. |
| `selected` | `list[str]` | ✅ | The diagnosis name strings the doctor selected. Must be non-empty. |

**Response Body**:

| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | Session ID. |
| `investigations` | `list[{name: str, reason: str}]` | Suggested investigations with a brief clinical rationale. |

**Errors**:
- `404` — session not found.
- `400` — `selected` list is empty.
- `500` / `503` — LLM failure.

---

### `POST /select-investigations`

**Description**: Records selected investigations and returns suggested medications.

**Request Body** (`SelectInvestigationsRequest`):

| Field | Type | Required | Description |
|---|---|---|---|
| `session_id` | `str` | ✅ | Session UUID. |
| `selected` | `list[str]` | ✅ | Investigation name strings selected. Non-empty required. |

**Response Body**:

| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | Session ID. |
| `medications` | `list[{name: str, dose: str, route: str}]` | Suggested medications with dose and route of administration. |

**Errors**: Same pattern as above.

---

### `POST /select-medications`

**Description**: Records selected medications and returns suggested clinical procedures.

**Request Body** (`SelectMedicationsRequest`):

| Field | Type | Required | Description |
|---|---|---|---|
| `session_id` | `str` | ✅ | Session UUID. |
| `selected` | `list[str]` | ✅ | Medication name strings. Non-empty required. |

**Response Body**:

| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | Session ID. |
| `procedures` | `list[{name: str, indication: str}]` | Suggested procedures with clinical indication. |

**Errors**: Same pattern as above.

---

## 9. Step-by-Step Flow

**Example patient**: Child (patient ID `791427b4-…`), presenting complaint: `"fever with cough"`.

---

**Step 1 — `POST /start`**

```json
// Request
{
  "patient_id": "791427b4-9cc4-8bcc-3fee-e3e14b6d3fea",
  "chief_complaint": "fever with cough",
  "clinical_history": "5-year-old boy, no known allergies"
}
```

- `_fetch_patient_context()` calls medical API: `GET /api/v1/patient/791427b4…/complaint/latest?complaint=fever+with+cough`
- Medical API returns: last visit date, diagnosis history, follow-up history (e.g. 2 previous visits with URTIs).
- `create_session()` is called; session stores complaint, history, follow-up history, current consultation.
- `build_question_prompt(session)` is called → LLM generates first question.
- GBNF grammar ensures valid JSON output.
- `validate_question()` confirms it has 2–6 options.

```json
// Response
{
  "session_id": "d2f4a8b1-...",
  "question_number": 1,
  "total_questions": 5,
  "question": "Fever since?",
  "options": ["1 day", "2 days", "3 days", "4+ days"],
  "completed": false
}
```

---

**Steps 2–5 — `POST /answer` × 4 more**

The clinician picks options for each question. Session accumulates:
- Q1: "Fever since?" → "3 days"
- Q2: "Peak temperature?" → "102°F"
- Q3: "Cough type?" → "Wet/Productive"
- Q4: "Breathing difficulty?" → "No"
- Q5: "Activity level?" → "Reduced"

After the 5th answer, `/answer` response includes `"completed": true`.

LLM prompt (sent for diagnosis) includes:
- System prompt instructing 6–10 ranked differential diagnoses
- Chief complaint, clinical history
- Current consultation data (vitals, key Q&A from last visit)
- Follow-up history (previous 2 visits with outcomes)
- All 5 Q&A pairs

```json
// Diagnosis response
{
  "session_id": "d2f4a8b1-...",
  "completed": true,
  "red_flag": false,
  "considerations": [
    {"name": "Viral Upper Respiratory Tract Infection", "likelihood": "high"},
    {"name": "Acute Bronchitis", "likelihood": "high"},
    {"name": "Pneumonia", "likelihood": "medium"},
    {"name": "Influenza", "likelihood": "medium"},
    {"name": "Allergic Rhinitis with Post-nasal Drip", "likelihood": "low"}
  ]
}
```

---

**Step 3 — `POST /select-diagnoses`**

Doctor selects "Viral Upper Respiratory Tract Infection" and "Acute Bronchitis".

LLM receives all prior context plus the two selected diagnoses and the 5 Q&A pairs, and returns:

```json
{
  "session_id": "d2f4a8b1-...",
  "investigations": [
    {"name": "Full Blood Count (FBC)", "reason": "To check for leukocytosis suggesting bacterial infection"},
    {"name": "Chest X-Ray", "reason": "To rule out pneumonia given productive cough"},
    {"name": "Throat Swab Culture", "reason": "To identify bacterial pathogens if antibiotic therapy considered"}
  ]
}
```

---

**Step 4 — `POST /select-investigations`**

Doctor selects FBC and Chest X-ray.

LLM generates medication suggestions. Context sent includes: complaint, history, current consultation, follow-up history, diagnoses, investigations selected.

```json
{
  "session_id": "d2f4a8b1-...",
  "medications": [
    {"name": "Paracetamol", "dose": "15 mg/kg every 6 hours", "route": "Oral"},
    {"name": "Salbutamol (Albuterol)", "dose": "2.5 mg nebulised every 4-6 hours as needed", "route": "Inhalation"},
    {"name": "Saline Nasal Drops", "dose": "2-3 drops each nostril, 3-4 times daily", "route": "Nasal"}
  ]
}
```

---

**Step 5 — `POST /select-medications`**

Doctor selects Paracetamol and Salbutamol.

LLM generates procedures. Context now includes all of the above plus the selected medications.

```json
{
  "session_id": "d2f4a8b1-...",
  "procedures": [
    {"name": "Nebulisation with Salbutamol", "indication": "Bronchospasm with wheeze on auscultation"},
    {"name": "Oxygen Saturation Monitoring", "indication": "Productive cough with reduced activity, monitor for desaturation"},
    {"name": "Dietary and Hydration Advice", "indication": "Promote oral fluid intake and rest for recovery"}
  ]
}
```

**Summary page** then displays all doctor-confirmed selections. Can be printed.

---

## 10. Prompt Design

All prompts use the **ChatML format** compatible with Qwen and aligned model conventions. The assistant turn is left open so llama.cpp generates the continuation. Grammar-constrained decoding is applied at the token level.

---

### Prompt 1 — `build_question_prompt(session)`

**Purpose**: Generate the next triage question.

**System instruction**: "You are a clinical triage assistant. Generate ONE short follow-up question with answer options that helps narrow the diagnosis."

**Format rules injected**:
- Question: 2–6 words ending with `?`
- Options: 1–5 word labels, 2–6 count depending on question type (Yes/No → 2, Duration → 4–5, Temperature → 5, Severity → 3–4, Category → 3–5)
- Must not repeat a previously asked question

**Clinical context injected** (from session):
- `chief_complaint`
- `clinical_history`
- Current consultation (formatted one-liner: visit date, vitals, key Q&A, diagnoses, medications, advice)
- Follow-up history (multi-line: each prior visit's date, complaints, diagnoses, medications, investigations, advice)
- All Q&A pairs already collected this session

**What LLM returns**: `{"question": "...", "options": ["...", ...]}`

---

### Prompt 2 — `build_diagnosis_prompt(session)`

**Purpose**: Generate ranked differential diagnoses after all 5 questions.

**System instruction**: "List 6–10 differential diagnoses ranked most to least likely. Use only 'high', 'medium', or 'low' for likelihood. Common conditions first."

**Clinical context injected**:
- `chief_complaint`, `clinical_history`
- Current consultation, follow-up history (same formatters)
- All 5 Q&A pairs formatted as `Q → A`

**What LLM returns**: `{"considerations": [{"name": "...", "likelihood": "high|medium|low"}, ...]}`

> **Note**: The `build_diagnosis_prompt` function in the current code is missing its `return` statement (line 123 goes directly to the end of the function without `return prompt`). This is a bug — the prompt string is created but not returned. The function implicitly returns `None`, meaning `prompt_fn(session)` in `_generate()` would be `None`, which would then be passed to `raw_fn(None)`. This would likely cause an error at the LLM call stage. This needs verification and fixing.

---

### Prompt 3 — `build_investigations_prompt(session)`

**Purpose**: Suggest relevant investigations for the selected diagnoses.

**System instruction**: "Suggest 5–10 relevant medical investigations. For each provide a short reason (why it is needed). Be specific and practical."

**Clinical context injected**:
- `chief_complaint`, `clinical_history`
- Current consultation, follow-up history
- All 5 Q&A pairs
- `selected_diagnoses` (comma-separated list of doctor-confirmed diagnoses)

**What LLM returns**: `{"investigations": [{"name": "...", "reason": "..."}, ...]}`

---

### Prompt 4 — `build_medications_prompt(session)`

**Purpose**: Suggest medications appropriate for the diagnoses and investigation findings.

**System instruction**: "Suggest 5–10 medications appropriate for the diagnoses and investigation findings. For each provide the typical adult dose and route of administration."

**Clinical context injected**:
- `chief_complaint`, `clinical_history`
- Current consultation, follow-up history
- `diagnoses` (comma-separated)
- `investigations` (comma-separated list of doctor-selected investigations)

> **Note**: The Q&A pairs are not injected into the medications prompt. Only diagnoses and investigations are provided alongside complaint, history, and patient context.

**What LLM returns**: `{"medications": [{"name": "...", "dose": "...", "route": "..."}, ...]}`

---

### Prompt 5 — `build_procedures_prompt(session)`

**Purpose**: Suggest clinical procedures or interventions for this patient's complete management plan.

**System instruction**: "Suggest 3–8 clinical procedures or interventions indicated for this patient. For each procedure give a clear indication (why it is needed for this patient)."

**Clinical context injected**:
- `chief_complaint`, `clinical_history`
- Current consultation, follow-up history
- `diagnoses` (comma-separated)
- `investigations` (comma-separated, doctor-selected)
- `medications` (comma-separated, doctor-selected)

**What LLM returns**: `{"procedures": [{"name": "...", "indication": "..."}, ...]}`

---

## 11. Error Handling and Fallbacks

### LLM Unreachable (Connection Error)

- `llm.py._call()` catches `requests.exceptions.ConnectionError` → raises `RuntimeError("Cannot reach Qwen…")`.
- `_generate()` catches `RuntimeError` from the LLM call → raises `HTTP 503` with the error message as detail.
- The frontend catches the 503 and calls `showErr(e.message)` — a dismissing error banner appears for 9 seconds.

### LLM Timeout

- `_call()` catches `requests.exceptions.Timeout` → raises `RuntimeError("Qwen timed out…")` → HTTP 503 path same as above.

### LLM Returns Bad JSON (Parse Failure)

- `parse_json()` returns `None` if the text cannot be parsed.
- `validate_fn(None)` returns `False`.
- `_generate()` logs a warning and retries up to `RETRY_COUNT` (5) times.
- If all 5 retries fail → HTTP 500: `"Could not generate valid {label} after 5 attempts."`

### LLM Returns Structurally Invalid Data (Validation Failure)

- Even if JSON parses, the `validate_*()` functions check field presence, types, and value constraints.
- Example: diagnosis with `likelihood = "uncertain"` → `validate_diagnosis()` returns `False` → retry.
- Same retry/500 path as above.

### Duplicate Questions

- `/answer` has a separate loop (also `RETRY_COUNT` iterations) that checks `q["question"] not in session["questions"]`.
- If the LLM keeps generating the same question text → HTTP 500: `"Could not generate a unique question."`

### Session Not Found

- `/answer`, `/select-diagnoses`, `/select-investigations`, `/select-medications` all call `get_session()`.
- If it returns `None` → HTTP 404: `"Session not found."` (Sessions are lost on server restart.)

### Empty Selection

- `/select-diagnoses`, `/select-investigations`, `/select-medications` check `if not req.selected` → HTTP 400: `"Select at least one diagnosis/investigation/medication."`

### Medical Backend Unavailable

- `_fetch_patient_context()` wraps everything in a bare `except Exception` → returns `({}, [])`.
- The session is created with empty `current_consultation` and `follow_up_history`.
- Prompts will show `"None"` for both fields.
- The triage proceeds without patient history context — no error is raised to the clinician.
- This is a silent degradation, not a hard failure.

### Medical Backend Returns 404

- Treated as "no prior history exists for this patient" — returns `({}, [])` same as above.

### Frontend Network/Fetch Error

- The `api()` helper in `index.html` catches all fetch errors and calls `showErr(error.message)`.
- The UI reverts to showing the last active section (question, diagnosis, etc.) and re-enables buttons.

### AI Notes Stream Failure (port 8004)

- If the SSE stream to the autocomplete service fails, the error is shown via `showErr("Failed to generate notes: …")`.
- The "Generate AI Notes" button shows "Failed - Retry" for 3 seconds, then resets to "Generate AI Notes".
- Notes textarea is simply left empty.

---

## 12. Known Limitations and Roadmap

### Known Limitations

| Issue | Location | Details |
|---|---|---|
| **Missing `return` in `build_diagnosis_prompt()`** | `prompts.py:123` | The diagnosis prompt string `prompt` is built but never returned. The function returns `None` implicitly. This will cause `_generate()` to pass `None` as the prompt to the LLM, which will likely raise an error or produce unexpected behavior. **This is a bug.** |
| **Hardcoded IP addresses everywhere** | `config.py`, `index.html` | `QWEN_URL`, `MEDICAL_API_BASE`, `BASE_TRIAGE`, `BASE_AUTOCOMPLETE` are all hardcoded to `34.180.37.249`. No environment variable support. Cannot deploy to a different machine without code edits. |
| **Hardcoded PATIENT_ID in frontend** | `index.html:690` | The frontend sends a fixed UUID (`791427b4-…`) as `patient_id` for every session. Real multi-patient use requires a login/patient selection flow. |
| **In-memory sessions only** | `state.py` | Sessions are not persisted. A server restart or process crash wipes all active sessions. No Redis or DB backing. |
| **No session expiry** | `state.py` | Sessions accumulate indefinitely in memory. In production, memory would leak if not bounded. |
| **Single-process** | Architecture | `uvicorn main:app` with one worker means sessions are in-process. Running multiple uvicorn workers would make sessions unavailable across workers. |
| **No `requirements.txt`** | Root | There is no dependency file. Reproduction requires manually reading `Read_Me` and install instructions. |
| **Red flag detection is keyword-only** | `main.py:57–61` | A simple substring match on answers — no NLP or clinical ontology. Phrases like "not breathing with difficulty" would be incorrectly flagged. |
| **Medications prompt omits Q&A context** | `prompts.py:build_medications_prompt` | The 5 Q&A pairs are not included in the medications prompt, unlike question/diagnosis/investigations prompts. |
| **AI Notes service not in this repo** | `index.html:BASE_AUTOCOMPLETE` | The SSE streaming notes generation at port 8004 is a completely separate service with no code in this repository. |
| **No authentication** | `main.py` | The API has no authentication. `allow_origins=["*"]` is used. Not suitable for production medical data handling. |
| **`temperature=0.3`; no seed** | `config.py`, `llm.py` | Responses are not deterministic. The same prompt may produce different results — helpful for retries, but makes testing unpredictable. |
| **Grammar currently applied to ALL tasks** | `llm.py` | `Benchmark.py` was written to test whether grammar slows down question generation, and found plain JSON prefill was faster. However, the current code still uses grammar for all 5 tasks. |

### Roadmap

- [ ] **Add `requirements.txt`** — pin all dependency versions.
- [ ] **Move all hardcoded IPs and secrets to `.env`** — use `python-dotenv` or environment variables.
- [ ] **Replace hardcoded `PATIENT_ID` in frontend** — implement a patient search or login screen.
- [ ] **Add Redis-backed session storage** — enable persistence, TTL, and multi-worker support.
- [ ] **Fix missing `return` in `build_diagnosis_prompt()`** — add `return prompt` on line 123.
- [ ] **Improve red-flag detection** — use NLP or a structured symptom ontology rather than substring matching.
- [ ] **Add question deduplication with semantic similarity** — current check is exact string match only.
- [ ] **Inject Q&A pairs into medications prompt** — for clinical completeness, consistent with other prompts.
- [ ] **Add basic authentication** — JWT or API key to protect the clinical endpoints.
- [ ] **Add structured logging/request tracing** — replace `print("[DEBUG] …")` calls with proper structured logs.
- [ ] **Add `/health` and `/ready` endpoints** — for deployment health checks.
- [ ] **Implement session timeout/cleanup** — a background task to purge sessions older than N hours.
- [ ] **Package as Docker container** — `Dockerfile` + `docker-compose.yml` to bundle the FastAPI app and llama.cpp server.
- [ ] **Create a proper `requirements.txt` generation step** — run `pip freeze > requirements.txt` after known good setup.

---

## 13. Changelog

The `.backup` files in the repository reveal the significant changes made between the original version and the current version.

### What Changed Between `*.backup` and Current

#### `main.py` — Added Patient Context Enrichment

| Change | Details |
|---|---|
| **Added `patient_id` to `StartRequest`** | The original `StartRequest` only had `chief_complaint` and `clinical_history`. The new version adds `patient_id: str` as a required field. |
| **Added `import requests`** | New dependency for calling the medical API. |
| **Added `MEDICAL_API_BASE` import from config** | The old version did not import or use this constant. |
| **Added `_fetch_patient_context()` function** | Entirely new function. Calls the medical API, handles 404 gracefully, extracts `follow_up_history` and the current consultation dict. Returns `({}, [])` on any error or 404. |
| **`/start` now fetches patient context before creating session** | The old `start()` went directly to `create_session(complaint, history)`. The new version first calls `_fetch_patient_context()`, then passes `follow_up_history` and `current_consultation` to `create_session()`. |
| **`create_session()` signature changed** | Now takes 4 arguments: `complaint`, `history`, `follow_up_history`, `current_consultation`. New session dict fields added. |
| **Added DEBUG print statements** | Extensive `print("[DEBUG]…")` logs throughout `_fetch_patient_context()` and `/start` tracking API calls, response codes, and session state. |

#### `prompts.py` — Added Follow-Up History and Consultation Context to All Prompts

| Change | Details |
|---|---|
| **Added `_format_follow_up_history()` private function** | New helper that formats the list of past visits into a readable multi-line string. Filters only `selected=True` items. |
| **Added `_format_current_consultation()` private function** | New helper that formats the most recent visit into a single summary line, including vitals (temp, BP, weight) and key Q&A pairs. |
| **All 5 prompt builders now inject patient history context** | Every prompt (question, diagnosis, investigations, medications, procedures) now includes `current consultation` and `Follow-up history` lines in the user section. The backup prompts had neither. |
| **`build_question_prompt` prints the full prompt** | Added `print("[DEBUG PROMPT]…")` calls that dump the full prompt string and the formatted history fields to stdout. |
| **`build_diagnosis_prompt` lost its `return` statement** | The original backup correctly had `return f"""…"""`. The new version assigns to `prompt=` but does not return it. This introduced a regression bug. |

#### `state.py` — Extended Session Schema

| Change | Details |
|---|---|
| **`create_session()` signature expanded** | Added `follow_up_history: list` and `current_consultation: dict` parameters with `or []` / `or {}` defaults. |
| **Session dict now stores `follow_up_history` and `current_consultation`** | Persisted on the session for use by all downstream prompt builders. |

#### `config.py` — Added Medical API Configuration

| Change | Details |
|---|---|
| **Added `MEDICAL_API_BASE`** | New constant pointing to the internal medical data service. |

#### `index.html` — Minor UI Additions

The current `index.html` is functionally very similar to `index.html.backup`. The main JavaScript logic (flow steps, API calls, rendering) is the same. The current version includes:
- The `patient_id` field sent with `/start` (hardcoded `PATIENT_ID` constant)
- Additional form field for `Vitals` (optional)
- Refined CSS (slightly different debug panel, notes sections, manual entry controls)

---

*This README was generated from reading the actual source code in the ClinicAssist-V2 repository. All details are derived from the code itself, not from assumptions.*





 to run qwen 

~/llama.cpp/build/bin/llama-server \
  -m /home/nathanivikas890_gmail_com/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --host 0.0.0.0 \
  --port 8006 \
  --ctx-size 4096
  
  uvicorn main:app --host 0.0.0.0 --port 9000 