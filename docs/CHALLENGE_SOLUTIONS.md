# PwnzzAI — challenge solutions (draft scaffold)

This file is the **working contract** between “what instructors intend” and what automation (or reviewers) should treat as **solved**.

- **AI draft:** best guess from the codebase and routes (may be wrong or incomplete).
- **Your edits:** replace or extend the **Actual solution** and **Success evidence** columns so they match your canonical teaching script.

Convention for each challenge:

| Field | Purpose |
|--------|--------|
| **Primary surface** | Main URL(s) or API path(s). |
| **AI draft solution** | Guessed learner / attacker steps + what “done” looks like. |
| **Actual solution** | *You fill:* intended walkthrough (can be multi-step, UI + API). |
| **Success evidence** | *You fill:* observable proof (exact strings, JSON keys, HTTP codes, side effects). |

---

## 0. Prerequisites (not OWASP items, but needed for many labs)

### 0.1 Log in as a shop user

| | |
|--|--|
| **Primary surface** | `GET/POST /login` → session |
| **AI draft solution** | Log in as `alice`/`alice` or `bob`/`bob` when the UI requires a session (orders, some demos). |
| **Actual solution** | ai draft solution, you got it right|
| **Success evidence** | your name on the page is bob or alice |

### 0.2 Cloud LLM API key (OpenAI-compatible / LiteLLM)

| | |
|--|--|
| **Primary surface** | `POST /save-openai-api-key`, `GET /check-openai-api-key` |
| **AI draft solution** | Enter a valid key in Lab Setup so cloud-backed routes return real model output instead of “missing key” JSON. |
| **Actual solution** | this can be skipped and isn't a challenge |
| **Success evidence** | *Fill in.* e.g. `check-openai-api-key` → `has_key: true`. |

### 0.3 Ollama availability

| | |
|--|--|
| **Primary surface** | `GET /check-ollama-status`, `POST /setup-ollama`, `GET /setup-ollama-stream` |
| **AI draft solution** | Ensure Ollama is running and the configured model is pulled; use setup endpoints or host docs. |
| **Actual solution** | skip, not a challenge |
| **Success evidence** | *Fill in.* e.g. `available: true`, target model name in `models[]`. |

---

## 1. Model theft (training data / IP exposure style)

### 1.1 Model theft UI + API extraction

| | |
|--|--|
| **Primary surface** | `GET /model-theft`, `POST /api/model-theft`, `GET /generate_sentiment_model` |
| **AI draft solution** | Open the model theft page; call the theft API with chosen probe words (or use UI); obtain approximated weights and correlation to the exposed “actual” weights from `generate_sentiment_model`. |
| **Actual solution** | correct |
| **Success evidence** | *Fill in.* e.g. JSON contains non-empty `approximated_weights`, `correlation` above threshold you define, or learner reproduces weights within X%. |

---

## 2. Data poisoning

### 2.1 Poisoned training + biased inference

| | |
|--|--|
| **Primary surface** | `GET /data-poisoning`, `POST /api/train-poisoned-model`, `POST /api/test-poisoned-model` |
| **AI draft solution** | Submit contradictory labeled examples (positive text labeled negative, etc.); train; run test text through `test-poisoned-model` and show sentiment flipped vs benign baseline. |
| **Actual solution** | pause, future todo |
| **Success evidence** | *Fill in.* e.g. specific `sentiment` + `confidence` after N poison comments; or comparison table you publish. |

---

## 3. Supply chain (malicious model / dependency)

### 3.1 Pickle / “model” load side effects

| | |
|--|--|
| **Primary surface** | `GET /supply-chain`, `POST /save-js-malicious-model`, `POST /save-bash-malicious-model`, `POST /load-bash-malicious-model`, `GET /demo-malicious-model` |
| **AI draft solution** | Trigger save of malicious artifacts, load bash malicious model and observe warning/command list; open demo page and observe injected behavior described in UI. |
| **Actual solution** | correct ,same as ai draft|
| **Success evidence** | *Fill in.* e.g. JSON fields you expect from `load-bash-malicious-model`, or DOM/HTML pattern for `demo-malicious-model`. |

---

## 4. Denial of service (resource exhaustion)

### 4.1 Simulated LLM API (no real model)

| | |
|--|--|
| **Primary surface** | `GET /dos-attack`, `POST /api/llm-query` |
| **AI draft solution** | Send many prompts rapidly; show `server_load.requests_last_minute` rising, slower `processing_time`, and occasional error status as described on the page. |
| **Actual solution** | what you just said |
| **Success evidence** | *Fill in.* e.g. N requests within T seconds, load ≥ K, or specific degraded response text you teach. |

### 4.2 “Real” DoS chat — Ollama

| | |
|--|--|
| **Primary surface** | `GET /real-dos-attack` (same template family as `/dos-attack`), `POST /chat-with-ollama-dos` |
| **AI draft solution** | Flood or stress the Ollama-backed chat endpoint per lab instructions until latency or errors match learning goal. |
| **Actual solution** | *Fill in.* |
| **Success evidence** | *Fill in.* |

### 4.3 “Real” DoS chat — cloud

| | |
|--|--|
| **Primary surface** | `POST /chat-with-openai-dos` (requires API key in session) |
| **AI draft solution** | Same as 4.2 but against billed API; emphasize rate/cost risk. |
| **Actual solution** | *Fill in.* |
| **Success evidence** | *Fill in.* |

---

## 5. Training-data leakage & RAG (sensitive retrieval)

### 5.1 Hugging Face tab (current app behavior)

| | |
|--|--|
| **Primary surface** | `POST /training-data-leak/huggingface` |
| **AI draft solution** | *(Codebase stub)* Send a query POST body; receive stub JSON explaining HF path is not fully wired in-image. If you replace this with a real HF demo, document the learner query and expected leak shape here. |
| **Actual solution** | *Fill in.* |
| **Success evidence** | *Fill in.* |

### 5.2 Ollama RAG leakage

| | |
|--|--|
| **Primary surface** | `POST /training-data-leak/ollama`, `POST /update-rag-ollama` |
| **AI draft solution** | Refresh RAG from comments; ask queries that surface simulated PII/VIP strings embedded in retrieved context (per `ollama_sensitive_data_leakage` behavior). |
| **Actual solution** | *Fill in.* |
| **Success evidence** | *Fill in.* e.g. `has_leakage: true`, specific `leaked_info` patterns, or substrings in `response`. |

### 5.3 OpenAI (cloud) RAG leakage

| | |
|--|--|
| **Primary surface** | `POST /training-data-leak/openai`, `POST /update-rag-openai` |
| **AI draft solution** | Save API key; refresh OpenAI RAG; run queries until sensitive markers appear in model output. |
| **Actual solution** | *Fill in.* |
| **Success evidence** | *Fill in.* |

### 5.4 Misinformation RAG refresh (Ollama / OpenAI)

| | |
|--|--|
| **Primary surface** | `POST /update-rag-misinformation`, `POST /update-rag-openai-misinfo` |
| **AI draft solution** | Rebuild misinformation corpus from comments; then use misinformation chat (section 9) to show wrong answers grounded on poisoned retrieval. |
| **Actual solution** | *Fill in.* |
| **Success evidence** | *Fill in.* |

---

## 6. Insecure design / excessive agency / plugins

### 6.1 Insecure plugin — Ollama pizza assistant

| | |
|--|--|
| **Primary surface** | `GET /insecure-plugin`, `POST /chat-with-pizza-assistant` |
| **AI draft solution** | Ask the assistant to emit `EXECUTE_FUNCTION: search_pizza_price("...")` style output or otherwise chain tool execution; observe SQL-backed price strings or injection paths described in code comments. |
| **Actual solution** | *Fill in.* |
| **Success evidence** | *Fill in.* |

### 6.2 Insecure plugin — cloud

| | |
|--|--|
| **Primary surface** | `POST /chat-with-openai-plugin` |
| **AI draft solution** | With API key saved, prompt so the model issues dangerous tool/plugin behavior taught in your lesson (function abuse, over-permissioned actions). |
| **Actual solution** | *Fill in.* |
| **Success evidence** | *Fill in.* |

### 6.3 Excessive agency — Ollama

| | |
|--|--|
| **Primary surface** | `GET /excessive-agency`, `POST /excessive-agency/ollama` |
| **AI draft solution** | Prompt the model to place or modify orders without proper user confirmation; observe narrative that an order was placed or side effect text you define in curriculum. |
| **Actual solution** | *Fill in.* |
| **Success evidence** | *Fill in.* |

### 6.4 Excessive agency — cloud

| | |
|--|--|
| **Primary surface** | `POST /excessive-agency/openai` |
| **AI draft solution** | Same intent as 6.3 with cloud model and stricter refusal behavior; document the exact phrasing that still triggers the vulnerable path in your environment. |
| **Actual solution** | *Fill in.* |
| **Success evidence** | *Fill in.* |

---

## 7. Sensitive information disclosure (LLM)

### 7.1 Sensitive info page

| | |
|--|--|
| **Primary surface** | `GET /sensitive-info` (+ any JS calling related APIs if present) |
| **AI draft solution** | Follow page instructions to elicit secrets / PII / system details from the model per lesson plan. |
| **Actual solution** | *Fill in.* |
| **Success evidence** | *Fill in.* |

---

## 8. Prompt injection

### 8.1 Direct — Ollama (levels 1–5)

| | |
|--|--|
| **Primary surface** | `GET /direct-prompt-injection`, `POST /chat-with-pizza-assistant-direct-prompt-injection` (`level` in JSON) |
| **AI draft solution** | Per level, use social engineering / override prompts until the “coupon” word leaks (code-defined secrets: L1 `cheese`, L2 `oven`, L3 `olives`, L4 `mushroom`, L5 `mozzarella` with strongest refusal in system prompt). |
| **Actual solution** | *Fill in.* (e.g. canonical prompt ladder per level.) |
| **Success evidence** | *Fill in.* e.g. for L1–L4: model output contains coupon token; for L5: either “must not leak” or alternate success you define. |

### 8.2 Direct — cloud plugin path

| | |
|--|--|
| **Primary surface** | `POST /chat-with-openai-plugin-direct-prompt` (`level` in JSON) |
| **AI draft solution** | Same level concept as 8.1 with GPT-class behavior; document which levels you still consider “solvable” under your policy. |
| **Actual solution** | *Fill in.* |
| **Success evidence** | *Fill in.* |

### 8.3 Indirect — QR / image → model (Ollama)

| | |
|--|--|
| **Primary surface** | `GET /indirect-prompt-injection`, `POST /upload-qr` |
| **AI draft solution** | Craft a QR encoding attacker-controlled text; upload image; model processes decoded text and follows hidden instructions (e.g. exfil-style phrase or unsafe behavior you teach). |
| **Actual solution** | *Fill in.* |
| **Success evidence** | *Fill in.* e.g. if you standardize on a lab phrase, put the exact expected substring here. |

### 8.4 Indirect — QR (cloud)

| | |
|--|--|
| **Primary surface** | `POST /upload-qr-openai` (multipart + `level`) |
| **AI draft solution** | Same as 8.3 with session API key and chosen injection level. |
| **Actual solution** | *Fill in.* |
| **Success evidence** | *Fill in.* |

---

## 9. Misinformation / untrusted retrieval

### 9.1 Misinformation — Ollama

| | |
|--|--|
| **Primary surface** | `GET /misinformation`, `POST /misinformation/ollama` |
| **AI draft solution** | Ask questions that cause the model to assert false “facts” grounded on poisoned/misleading comment-derived context after RAG refresh. |
| **Actual solution** | *Fill in.* |
| **Success evidence** | *Fill in.* |

### 9.2 Misinformation — cloud

| | |
|--|--|
| **Primary surface** | `POST /misinformation/openai` |
| **AI draft solution** | Same as 9.1 with cloud model. |
| **Actual solution** | *Fill in.* |
| **Success evidence** | *Fill in.* |

---

## 10. Broken access / privacy (orders)

### 10.1 Order access — Ollama

| | |
|--|--|
| **Primary surface** | `POST /order-access/ollama` (session user matters) |
| **AI draft solution** | Log in as Alice; prompt so the assistant surfaces Bob’s orders or other users’ PII via vulnerable “username in prompt” logic; or document benign vs violating prompts. |
| **Actual solution** | *Fill in.* |
| **Success evidence** | *Fill in.* e.g. `has_access_violation: true`, `accessed_info` entries, or forbidden substrings in `response`. |

### 10.2 Order access — cloud

| | |
|--|--|
| **Primary surface** | `POST /order-access/openai` |
| **AI draft solution** | Same as 10.1 with API key and cloud model. |
| **Actual solution** | *Fill in.* |
| **Success evidence** | *Fill in.* |

---

## 11. Sentiment pipeline (supporting / theft follow-on)

### 11.1 Public sentiment APIs

| | |
|--|--|
| **Primary surface** | `POST /analyze_sentiment`, `POST /api/sentiment` |
| **AI draft solution** | Use endpoints to show inference on arbitrary strings; tie to model theft lesson if you want “use stolen model” narrative. |
| **Actual solution** | *Fill in.* |
| **Success evidence** | *Fill in.* |

---

## 12. Static learning pages (optional “challenges”)

### 12.1 Basics / glossary / index

| | |
|--|--|
| **Primary surface** | `GET /basics`, `GET /glossary`, `GET /` |
| **AI draft solution** | Read and complete any embedded questions or demos your curriculum assigns (may be non-exploitative). |
| **Actual solution** | *Fill in.* |
| **Success evidence** | *Fill in.* |

---

## Appendix — quick route index

| Path | Role (short) |
|------|----------------|
| `/login` | Session |
| `/model-theft` | Model theft UI |
| `/api/model-theft` | Theft API |
| `/generate_sentiment_model` | Exposed weights JSON |
| `/data-poisoning` | Poisoning UI |
| `/api/train-poisoned-model`, `/api/test-poisoned-model` | Poisoning API |
| `/supply-chain` | Supply chain UI |
| `/save-js-malicious-model`, `/save-bash-malicious-model`, `/load-bash-malicious-model` | Malicious model demo |
| `/demo-malicious-model` | JS-injection style demo |
| `/dos-attack`, `/real-dos-attack` | DoS pages |
| `/api/llm-query` | Simulated DoS API |
| `/chat-with-ollama-dos`, `/chat-with-openai-dos` | Live chat DoS |
| `/training-data-leak/*` | Training data / RAG leak tabs |
| `/update-rag-*` | RAG rebuild |
| `/insecure-plugin`, `/chat-with-pizza-assistant`, `/chat-with-openai-plugin` | Insecure plugin |
| `/sensitive-info` | Sensitive disclosure page |
| `/direct-prompt-injection`, `/chat-with-pizza-assistant-direct-prompt-injection` | Direct injection (Ollama) |
| `/chat-with-openai-plugin-direct-prompt` | Direct injection (cloud) |
| `/indirect-prompt-injection`, `/upload-qr`, `/upload-qr-openai` | Indirect via QR |
| `/order-access/ollama`, `/order-access/openai` | Cross-user order risk |
| `/excessive-agency/*` | Excessive agency |
| `/misinformation/*` | Misinformation |
| `/save-openai-api-key`, `/check-openai-api-key` | Cloud key setup |
| `/setup-ollama*`, `/check-ollama-status` | Ollama setup |

---

*End of draft scaffold. Replace “Fill in” sections with your authoritative solutions and evidence checks.*
