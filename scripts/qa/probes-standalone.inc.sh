#!/usr/bin/env bash
# Shared standalone PwnzzAI + model-container probes (sourced by docker-smoke-test.sh and run-model-integration-gates.sh).
# Set APP_BASE to http://host:port (no trailing slash), then call run_standalone_model_probes.

require_app_base() {
  if [[ -z "${APP_BASE:-}" ]]; then
    echo "APP_BASE must be set (e.g. http://127.0.0.1:8080)" >&2
    exit 1
  fi
}

log_probe() { printf '[probe] %s\n' "$*" >&2; }

probe_app_root() {
  curl -fsS "${APP_BASE}/" >/dev/null
  log_probe "GET / OK"
}

probe_basics_page() {
  curl -fsS "${APP_BASE}/basics" >/dev/null
  log_probe "GET /basics OK"
}

probe_login_page() {
  curl -fsS "${APP_BASE}/login" >/dev/null
  log_probe "GET /login OK"
}

probe_ollama_status_json() {
  local body
  body=$(curl -fsS "${APP_BASE}/check-ollama-status")
  echo "$body" | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert 'available' in d, 'missing available'
assert 'models' in d, 'missing models'
assert isinstance(d['models'], list), 'models must be a list'
print(d.get('available'), len(d.get('models') or []))
" >/dev/null
  log_probe "GET /check-ollama-status JSON OK"
}

# If EXPECTED_OLLAMA_MODEL is set, require that tag when Ollama is available
probe_expected_model_tag() {
  [[ -z "${EXPECTED_OLLAMA_MODEL:-}" ]] && return 0
  local body
  body=$(curl -fsS "${APP_BASE}/check-ollama-status")
  export EXPECTED_OLLAMA_MODEL
  echo "$body" | python3 -c "
import json, sys, os
want = (os.environ.get('EXPECTED_OLLAMA_MODEL') or '').strip()
if not want:
    sys.exit(0)
d = json.load(sys.stdin)
if not d.get('available'):
    raise SystemExit('Ollama not available; cannot verify model tag')
models = d.get('models') or []
if want not in models:
    raise SystemExit('model %r not in %r; run: ollama pull %s' % (want, models, want))
"
  log_probe "EXPECTED_OLLAMA_MODEL=${EXPECTED_OLLAMA_MODEL} present"
}

probe_chat_ollama_dos() {
  local body
  body=$(curl -fsS -X POST "${APP_BASE}/chat-with-ollama-dos" \
    -H "Content-Type: application/json" \
    -d '{"message":"Say the word ok only."}')
  echo "$body" | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert 'response' in d, 'missing response key'
assert d['response'] is not None and str(d['response']).strip() != '', 'empty response'
"
  log_probe "POST /chat-with-ollama-dos OK"
}

probe_chat_ollama_dos_validation() {
  local code
  code=$(curl -sS -o /dev/null -w '%{http_code}' -X POST "${APP_BASE}/chat-with-ollama-dos" \
    -H "Content-Type: application/json" \
    -d '{"message":""}')
  [[ "$code" == "400" ]] || { log_probe "expected HTTP 400 for empty message, got ${code}"; return 1; }
  log_probe "POST /chat-with-ollama-dos empty body -> 400 OK"
}

run_standalone_model_probes() {
  require_app_base
  log_probe "APP_BASE=${APP_BASE}"
  probe_app_root
  probe_basics_page
  probe_login_page
  probe_ollama_status_json
  probe_expected_model_tag
  probe_chat_ollama_dos
  probe_chat_ollama_dos_validation
  log_probe "all standalone model probes passed"
}
