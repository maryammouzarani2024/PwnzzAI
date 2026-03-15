#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_IMAGE="${APP_IMAGE:-pwnzzai:local-smoke}"
EXT_OLLAMA_CONTAINER="ollama-ext-test"
EXT_OLLAMA_PORT="11435"
APP_PORT="8080"

log() {
  printf "\n[%s] %s\n" "$(date +"%H:%M:%S")" "$1"
}

wait_for_http() {
  local url="$1"
  local retries="${2:-30}"
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
  set +e
  log "Cleaning up test containers"
  (cd "$ROOT_DIR" && PWNZZAI_IMAGE="$APP_IMAGE" docker compose down >/dev/null 2>&1)
  (cd "$ROOT_DIR" && PWNZZAI_IMAGE="$APP_IMAGE" OLLAMA_HOST="http://host.docker.internal:${EXT_OLLAMA_PORT}" docker compose -f docker-compose.external-ollama.yml down >/dev/null 2>&1)
  docker rm -f "$EXT_OLLAMA_CONTAINER" >/dev/null 2>&1 || true
}

trap cleanup EXIT

log "Checking docker prerequisites"
command -v docker >/dev/null 2>&1 || { echo "docker is required"; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "docker compose plugin is required"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is required"; exit 1; }

log "Validating compose files"
(cd "$ROOT_DIR" && docker compose -f docker-compose.yml config >/dev/null)
(cd "$ROOT_DIR" && docker compose -f docker-compose.external-ollama.yml config >/dev/null)

log "Building local app image: ${APP_IMAGE}"
(cd "$ROOT_DIR" && docker build -t "$APP_IMAGE" .)

log "Testing Option 1: bundled compose (PwnzzAI + Ollama)"
(cd "$ROOT_DIR" && PWNZZAI_IMAGE="$APP_IMAGE" docker compose up -d)
if ! wait_for_http "http://localhost:${APP_PORT}" 45 2; then
  echo "App did not become reachable for Option 1"
  (cd "$ROOT_DIR" && docker compose logs --tail=200)
  exit 1
fi
log "Option 1 passed"
(cd "$ROOT_DIR" && PWNZZAI_IMAGE="$APP_IMAGE" docker compose down)

log "Starting standalone Ollama container for Option 2 test"
docker rm -f "$EXT_OLLAMA_CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$EXT_OLLAMA_CONTAINER" -p "${EXT_OLLAMA_PORT}:11434" ollama/ollama:latest >/dev/null

log "Testing Option 2: external Ollama compose"
(
  cd "$ROOT_DIR" && \
  PWNZZAI_IMAGE="$APP_IMAGE" \
  OLLAMA_HOST="http://host.docker.internal:${EXT_OLLAMA_PORT}" \
  docker compose -f docker-compose.external-ollama.yml up -d
)
if ! wait_for_http "http://localhost:${APP_PORT}" 45 2; then
  echo "App did not become reachable for Option 2"
  (cd "$ROOT_DIR" && docker compose -f docker-compose.external-ollama.yml logs --tail=200)
  exit 1
fi
log "Option 2 passed"

log "All smoke tests passed successfully"
