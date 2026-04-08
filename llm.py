import logging
import requests
from config import (
    QWEN_URL, TEMPERATURE,
    MAX_TOKENS_QUESTION, MAX_TOKENS_DIAGNOSIS,
    MAX_TOKENS_INVESTIGATIONS, MAX_TOKENS_MEDICATIONS, MAX_TOKENS_PROCEDURES,
    TIMEOUT_QUESTION, TIMEOUT_DIAGNOSIS,
    TIMEOUT_INVESTIGATIONS, TIMEOUT_MEDICATIONS, TIMEOUT_PROCEDURES,
)
from grammar import (
    QUESTION_GRAMMAR, DIAGNOSIS_GRAMMAR,
    INVESTIGATIONS_GRAMMAR, MEDICATIONS_GRAMMAR, PROCEDURES_GRAMMAR,
)

logger = logging.getLogger(__name__)


def _call(prompt: str, grammar: str, max_tokens: int, timeout: int) -> str:
    try:
        r = requests.post(
            QWEN_URL,
            json={
                "prompt":      prompt,
                "n_predict":   max_tokens,
                "temperature": TEMPERATURE,
                "stop":        ["</s>", "<|im_end|>"],
                "grammar":     grammar,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        content = r.json()["content"]
        logger.debug("[LLM raw] %s", content)
        return content

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach Qwen at {QWEN_URL}. "
            "Start llama.cpp on the configured completion port (8006 by default)."
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Qwen timed out after {timeout}s. "
            "Server may be overloaded — restart llama.cpp and try again."
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Qwen HTTP error: {e}")
    except (KeyError, ValueError):
        raise RuntimeError("Unexpected response from Qwen (missing 'content').")


def generate_question_raw(prompt: str)       -> str:
    return _call(prompt, QUESTION_GRAMMAR,       MAX_TOKENS_QUESTION,       TIMEOUT_QUESTION)

def generate_diagnosis_raw(prompt: str)      -> str:
    return _call(prompt, DIAGNOSIS_GRAMMAR,      MAX_TOKENS_DIAGNOSIS,      TIMEOUT_DIAGNOSIS)

def generate_investigations_raw(prompt: str) -> str:
    return _call(prompt, INVESTIGATIONS_GRAMMAR, MAX_TOKENS_INVESTIGATIONS, TIMEOUT_INVESTIGATIONS)

def generate_medications_raw(prompt: str)    -> str:
    return _call(prompt, MEDICATIONS_GRAMMAR,    MAX_TOKENS_MEDICATIONS,    TIMEOUT_MEDICATIONS)

def generate_procedures_raw(prompt: str)     -> str:
    return _call(prompt, PROCEDURES_GRAMMAR,     MAX_TOKENS_PROCEDURES,     TIMEOUT_PROCEDURES)