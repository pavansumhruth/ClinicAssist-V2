# GBNF grammars for llama.cpp constrained decoding.
# The grammar is passed as "grammar" in the request body.
# The model can ONLY output tokens that match — no preamble,
# no markdown, no malformed JSON is possible.

# ── Question ─────────────────────────────────────────────────
# {"question": "...", "options": ["...", ...]}   2-6 options
QUESTION_GRAMMAR = r"""
root    ::= "{" ws "\"question\"" ws ":" ws string ws "," ws "\"options\"" ws ":" ws options ws "}"
options ::= "[" ws string (ws "," ws string){1,5} ws "]"
string  ::= "\"" char* "\""
char    ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
ws      ::= [ \t\n]*
"""

# ── Diagnosis ─────────────────────────────────────────────────
# {"considerations": [{"name": "...", "likelihood": "high|medium|low"}, ...]}
DIAGNOSIS_GRAMMAR = r"""
root       ::= "{" ws "\"considerations\"" ws ":" ws "[" ws item (ws "," ws item){1,9} ws "]" ws "}"
item       ::= "{" ws "\"name\"" ws ":" ws string ws "," ws "\"likelihood\"" ws ":" ws likelihood ws "}"
likelihood ::= "\"high\"" | "\"medium\"" | "\"low\""
string     ::= "\"" char* "\""
char       ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
ws         ::= [ \t\n]*
"""

# ── Investigations ────────────────────────────────────────────
# {"investigations": [{"name": "...", "reason": "..."}, ...]}
INVESTIGATIONS_GRAMMAR = r"""
root  ::= "{" ws "\"investigations\"" ws ":" ws "[" ws item (ws "," ws item){1,9} ws "]" ws "}"
item  ::= "{" ws "\"name\"" ws ":" ws string ws "," ws "\"reason\"" ws ":" ws string ws "}"
string ::= "\"" char* "\""
char   ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
ws     ::= [ \t\n]*
"""

# ── Medications ───────────────────────────────────────────────
# {"medications": [{"name": "...", "dose": "...", "route": "..."}, ...]}
MEDICATIONS_GRAMMAR = r"""
root  ::= "{" ws "\"medications\"" ws ":" ws "[" ws item (ws "," ws item){1,9} ws "]" ws "}"
item  ::= "{" ws "\"name\"" ws ":" ws string ws "," ws "\"dose\"" ws ":" ws string ws "," ws "\"route\"" ws ":" ws string ws "}"
string ::= "\"" char* "\""
char   ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
ws     ::= [ \t\n]*
"""

# ── Procedures ────────────────────────────────────────────────
# {"procedures": [{"name": "...", "indication": "..."}, ...]}
PROCEDURES_GRAMMAR = r"""
root  ::= "{" ws "\"procedures\"" ws ":" ws "[" ws item (ws "," ws item){1,9} ws "]" ws "}"
item  ::= "{" ws "\"name\"" ws ":" ws string ws "," ws "\"indication\"" ws ":" ws string ws "}"
string ::= "\"" char* "\""
char   ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
ws     ::= [ \t\n]*
"""