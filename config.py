# ── Server ──────────────────────────────────────────────────
QWEN_URL = "http://localhost:8006/completion"

# ── LLM token limits (separate per task) ────────────────────
MAX_TOKENS_QUESTION      = 128
MAX_TOKENS_DIAGNOSIS     = 350
MAX_TOKENS_INVESTIGATIONS = 400
MAX_TOKENS_MEDICATIONS   = 400
MAX_TOKENS_PROCEDURES    = 350

# ── LLM other settings ──────────────────────────────────────
TEMPERATURE  = 0.3
RETRY_COUNT  = 5

# ── Request timeouts ─────────────────────────────────────────
TIMEOUT_QUESTION      = 30
TIMEOUT_DIAGNOSIS     = 60
TIMEOUT_INVESTIGATIONS = 120
TIMEOUT_MEDICATIONS   = 120
TIMEOUT_PROCEDURES    = 60

# ── Session ──────────────────────────────────────────────────
MAX_QUESTIONS = 5
