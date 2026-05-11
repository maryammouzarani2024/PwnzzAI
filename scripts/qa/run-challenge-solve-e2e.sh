#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_PORT="${APP_PORT:-8080}"
APP_BASE="${APP_BASE:-http://127.0.0.1:${APP_PORT}}"
# Must match the tag the app uses (see docker-compose OLLAMA_MODEL) so Ollama has the model loaded.
E2E_OLLAMA_MODEL="${E2E_OLLAMA_MODEL:-llama3.2:1b}"
export OLLAMA_MODEL="${E2E_OLLAMA_MODEL}"
PWNZZAI_IMAGE="${PWNZZAI_IMAGE:-pwnzzai:e2e-local}"
KEEP_STACK_UP="${KEEP_STACK_UP:-0}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

wait_for_http() {
  local url="$1"
  local retries="${2:-60}"
  local sleep_seconds="${3:-2}"
  local i
  for ((i=1; i<=retries; i++)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$sleep_seconds"
  done
  return 1
}

cleanup() {
  if [[ "${KEEP_STACK_UP}" == "1" ]]; then
    log "KEEP_STACK_UP=1, leaving stack running."
    return
  fi
  log "Shutting down compose stack"
  (cd "${ROOT_DIR}" && PWNZZAI_IMAGE="${PWNZZAI_IMAGE}" docker compose down >/dev/null 2>&1 || true)
}

trap cleanup EXIT

command -v docker >/dev/null 2>&1 || { echo "docker is required"; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "docker compose plugin is required"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is required"; exit 1; }

if [[ ! -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  echo "Expected virtualenv python at ${ROOT_DIR}/.venv/bin/python"
  echo "Create it first, e.g. python3 -m venv .venv && .venv/bin/pip install -r requirements-test.txt"
  exit 1
fi

log "Building app image ${PWNZZAI_IMAGE}"
(cd "${ROOT_DIR}" && docker build -t "${PWNZZAI_IMAGE}" .)

log "Starting compose stack"
(cd "${ROOT_DIR}" && PWNZZAI_IMAGE="${PWNZZAI_IMAGE}" docker compose up -d)

log "Waiting for app: ${APP_BASE}"
if ! wait_for_http "${APP_BASE}/" 90 2; then
  echo "App did not become reachable at ${APP_BASE}"
  (cd "${ROOT_DIR}" && docker compose logs --tail=250)
  exit 1
fi

log "Waiting for ollama status endpoint"
if ! wait_for_http "${APP_BASE}/check-ollama-status" 90 2; then
  echo "Ollama status endpoint did not become reachable"
  (cd "${ROOT_DIR}" && docker compose logs --tail=250)
  exit 1
fi

log "Ensuring Ollama model is pulled: ${OLLAMA_MODEL}"
docker exec ollama ollama pull "${OLLAMA_MODEL}" >/dev/null

log "Running E2E solvability harness (set E2E_OPENAI_API_KEY for cloud-marked tests; E2E_SKIP_RAG_REFRESH=1 to skip slow RAG refresh)"
RUN_E2E=1 APP_BASE="${APP_BASE}" OLLAMA_MODEL="${OLLAMA_MODEL}" "${ROOT_DIR}/.venv/bin/python" -m pytest \
  "${ROOT_DIR}/tests/e2e/test_challenge_solvability_e2e.py" -q

log "E2E solvability harness passed."
