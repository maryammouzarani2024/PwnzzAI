# RFC-EX-B: Direct Prompt Injection

## Decision
`skip`

## Why
- PwnzzAI already ships the same risk class and attacker objective: user-controlled text is concatenated into an LLM chat against a system prompt that embeds a secret coupon word per difficulty level.
- **Duplicate rule:** attacker objective + risk class + exploit mechanic already exist at `GET /direct-prompt-injection`, `POST /chat-with-pizza-assistant-direct-prompt-injection`, and `POST /chat-with-openai-plugin-direct-prompt`.

## Solution
- Treat the existing lab as the canonical in-app direct-injection surface.
- Lock a **machine-verifiable contract** in-repo:
  - Levelized endpoints accept JSON `{ "message", "level" }` and return `{ "response": ... }` on success.
  - Expected naive-exfiltration markers for levels 1–4 stay aligned with `tests/e2e/challenge_catalog.py` (`OLLAMA_DIRECT_LEVEL_SECRETS` / `OPENAI_DIRECT_LEVEL_SECRETS`).
  - Level 5 remains the intentional “hold the line” profile (non-leak goal validated in E2E, not substring match).
- Add **functional workflow tests** that mock LLM backends so CI proves the HTTP contract and “exploit evidence in model output → surfaced to client” without requiring a live Ollama/OpenAI call.
- Keep **live exploitation proof** in `tests/e2e/test_challenge_solvability_e2e.py` (retries, real model variance).

## What This Does
- Documents why no second direct-injection tile or parallel route set is added.
- Gives deterministic regression coverage for routing, JSON shape, empty-body errors, and mocked success/failure semantics for both Ollama and OpenAI surfaces.

## What This Does Not Do
- No new challenge page, secret vocabulary, or extra levels beyond the current 1–5 design.
- No in-app “hardened mode” toggle for this lab in this RFC (mitigation remains instructional: system/user isolation, allowlists, monitoring, safer tool routing).

## Flag Definition
- **Vulnerable-mode flag (levels 1–4):** assistant `response` text contains the level’s coupon substring (case-insensitive), after a direct user attempt to override safety and extract the secret.
- **Level 5:** success is “no stable substring leak of `mozzarella`” under the E2E benign and attack prompts (see E2E tests); functional tests use mocked “safe” assistant output.
- **OpenAI path:** same level→secret mapping when a session (or allowed env) API key is present.

## Implementation Ladder
- **B1 Scope:** confirm duplicate against `/direct-prompt-injection` and paired POST routes — `skip` for new product surface.
- **B2 Contract:** freeze JSON fields, status codes, and level→secret catalog linkage (shared with E2E catalog).
- **B3 Build:** ship `tests/functional/test_rfc_ex_b_direct_prompt_injection_workflow.py` with mocks.
- **B4 Verify:** `pytest tests/functional/test_rfc_ex_b_direct_prompt_injection_workflow.py`; E2E unchanged as optional live gate.
- **B5 Release:** this RFC + tests; no user-facing copy change required.

## Checkpoints
- Duplicate decision recorded.
- Ollama + OpenAI POST contracts covered under mock.
- Catalog alignment verified (levels 1–4 needles).

## Tests
- Functional: empty message 400, mocked exfil for levels 1–4, mocked safe level 5, OpenAI route with session key + mock.
- E2E (existing): Ollama/OpenAI direct injection level suite.
