"""
Run this on YOUR machine (where llama.cpp is running) to find safe timeout values.
Usage:  python3 benchmark.py

It will print recommended values to paste into config.py.
"""
import requests, time, statistics

from config import QWEN_URL

RUNS = 3   # number of test calls per category

def call(label, body, runs=RUNS):
    times = []
    for i in range(runs):
        t = time.time()
        try:
            r = requests.post(QWEN_URL, json=body, timeout=120)
            r.raise_for_status()
            elapsed = time.time() - t
            out = r.json().get("content", "")
            times.append(elapsed)
            print(f"  run {i+1}: {elapsed:.2f}s  →  {out[:60]!r}")
        except Exception as e:
            print(f"  run {i+1}: FAILED — {e}")
    if times:
        avg = statistics.mean(times)
        worst = max(times)
        # recommend timeout = worst * 2.5, rounded up to nearest 5
        rec = max(10, int((worst * 2.5 / 5 + 1)) * 5)
        print(f"  avg={avg:.2f}s  worst={worst:.2f}s  → recommended timeout: {rec}s\n")
        return rec
    return 30

GRAMMAR = r"""
root    ::= "{" ws "\"question\"" ws ":" ws string ws "," ws "\"options\"" ws ":" ws options ws "}"
options ::= "[" ws string (ws "," ws string){1,5} ws "]"
string  ::= "\"" char* "\""
char    ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
ws      ::= [ \t\n]*
"""

print("=" * 55)
print("Qwen benchmark — runs each test 3 times")
print("=" * 55)

print("\n[1] Question WITH grammar (80 tokens)")
t_q_grammar = call("q+grammar", {
    "prompt": (
        "<|im_start|>system\nClinical triage. ONE question 1-4 words ending ?. "
        "Options: short labels.\n<|im_end|>\n"
        "<|im_start|>user\nComplaint: fever with cough\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "n_predict": 80, "temperature": 0.1,
    "stop": ["</s>", "<|im_end|>"],
    "grammar": GRAMMAR,
})

print("[2] Question WITHOUT grammar, prefill { (80 tokens)")
t_q_plain = call("q-plain", {
    "prompt": (
        "<|im_start|>system\nOutput ONLY JSON. No text before or after.\n<|im_end|>\n"
        "<|im_start|>user\nComplaint: fever with cough. ONE triage question, 1-4 words ending ?, "
        "with short answer options.\n<|im_end|>\n"
        "<|im_start|>assistant\n{"
    ),
    "n_predict": 80, "temperature": 0.1,
    "stop": ["</s>", "<|im_end|>"],
})

print("[3] Diagnosis (280 tokens, no grammar)")
t_diag = call("diagnosis", {
    "prompt": (
        "<|im_start|>system\nOutput ONLY JSON. No text before or after.\n<|im_end|>\n"
        "<|im_start|>user\nComplaint: fever with cough | QA: Fever since?:3 days | "
        "List 6-8 diagnoses.\n<|im_end|>\n"
        "<|im_start|>assistant\n{"
    ),
    "n_predict": 280, "temperature": 0.1,
    "stop": ["</s>", "<|im_end|>"],
})

print("[4] Investigations (320 tokens, no grammar)")
t_inv = call("investigations", {
    "prompt": (
        "<|im_start|>system\nOutput ONLY JSON. No text before or after.\n<|im_end|>\n"
        "<|im_start|>user\nComplaint: fever | Diagnoses: Viral fever, URTI | "
        "List 5-8 investigations.\n<|im_end|>\n"
        "<|im_start|>assistant\n{"
    ),
    "n_predict": 320, "temperature": 0.1,
    "stop": ["</s>", "<|im_end|>"],
})

print("=" * 55)
print("Paste these into config.py:")
print(f"  TIMEOUT_QUESTION       = {t_q_plain}")
print(f"  TIMEOUT_DIAGNOSIS      = {t_diag}")
print(f"  TIMEOUT_INVESTIGATIONS = {t_inv}")
print(f"  TIMEOUT_MEDICATIONS    = {t_inv}   # same profile as investigations")
print(f"  TIMEOUT_PROCEDURES     = {t_diag}  # slightly shorter output")
print()
print(f"  Grammar question was: {t_q_grammar}s timeout")
print(f"  Plain question was:   {t_q_plain}s timeout")
if t_q_grammar > t_q_plain:
    print("  ✓ Plain is faster — grammar removed from questions too")
