# ── Server ──────────────────────────────────────────────────
QWEN_URL = "http://34.14.197.45:8006/completion"
MEDICAL_API_BASE = "http://34.14.197.45:8000" 
SEARCH_API_BASE = "http://34.14.197.45:8003"
# ── LLM token limits (separate per task) ────────────────────
MAX_TOKENS_QUESTION      = 128
MAX_TOKENS_DIAGNOSIS     = 300
MAX_TOKENS_INVESTIGATIONS = 600
MAX_TOKENS_MEDICATIONS   = 600
MAX_TOKENS_PROCEDURES    = 600

# ── LLM other settings ──────────────────────────────────────
TEMPERATURE  = 0.3
RETRY_COUNT  = 5

# ── Request timeouts ─────────────────────────────────────────
TIMEOUT_QUESTION      = 90
TIMEOUT_DIAGNOSIS     = 150
TIMEOUT_INVESTIGATIONS = 180
TIMEOUT_MEDICATIONS   = 200
TIMEOUT_PROCEDURES    = 200

# ── Session ──────────────────────────────────────────────────
MAX_QUESTIONS = 5
