#!/usr/bin/env bash
# Sourced by setup-model-and-ctfd.sh — validate no port conflicts and one shared Ollama for the workshop.
# Requires: COMPOSE_FILE, ROOT_DIR, MODEL_HOST, OLLAMA_HOST (optional, for messaging)

log_val() { printf '[%s] [validate] %s\n' "$(date -Is)" "$*" >&2; }

# Workshop compose starts the ollama service only when profile local-ollama is enabled (default).
using_local_ollama_profile() {
  case "${COMPOSE_PROFILES:-local-ollama}" in
    *local-ollama*) return 0 ;;
    *) return 1 ;;
  esac
}

# True if MODEL_HOST points at this machine's published Ollama (compose binds 11434).
is_local_model_host() {
  case "${MODEL_HOST:-}" in
    http://127.0.0.1:11434|http://localhost:11434|http://0.0.0.0:11434) return 0 ;;
    *) return 1 ;;
  esac
}

# Before bringing the stack up: avoid conflicts on :11434 unless it already looks like Ollama.
check_port_11434_free_or_warn() {
  using_local_ollama_profile || return 0
  is_local_model_host || return 0
  if ! command -v ss >/dev/null 2>&1; then
    return 0
  fi
  if ! ss -tln 2>/dev/null | grep -qE ':(11434)\s'; then
    return 0
  fi
  if curl -fsS --max-time 3 "${MODEL_HOST}/api/tags" >/dev/null 2>&1; then
    log_val "Port 11434 already exposes an Ollama-compatible API; compose will reuse or replace the workshop ollama service."
    return 0
  fi
  log_val "Port 11434 is in use but does not respond like Ollama at ${MODEL_HOST}/api/tags."
  log_val "Free the port, or set OLLAMA_HOST / MODEL_HOST to a dedicated model host (separate machine or IP)."
  if [[ "${ALLOW_PORT_11434_CONFLICT:-}" == "1" ]]; then
    log_val "ALLOW_PORT_11434_CONFLICT=1 — continuing anyway."
    return 0
  fi
  return 1
}

# After compose up: exactly one running container for service ollama in this project.
verify_single_ollama_service() {
  if ! using_local_ollama_profile; then
    log_val "COMPOSE_PROFILES has no local-ollama — no local ollama container (use remote OLLAMA_HOST)."
    return 0
  fi
  local out
  out=$(docker compose -f "$COMPOSE_FILE" --project-directory "$(dirname "$COMPOSE_FILE")" ps -q ollama 2>/dev/null || true)
  if [[ -z "$out" ]]; then
    log_val "No running 'ollama' service found for this compose file."
    return 1
  fi
  local n
  n=$(echo "$out" | wc -l | tr -d ' ')
  if [[ "$n" -gt 1 ]]; then
    log_val "Expected one ollama container, found ${n}."
    return 1
  fi
  log_val "Single shared ollama service is running (one model container for all Pwnzai instances)."
  return 0
}

# Best-effort GPU visibility inside the ollama container (requires NVIDIA Container Toolkit on host).
verify_gpu_in_ollama_optional() {
  if ! using_local_ollama_profile; then
    return 0
  fi
  if ! is_local_model_host; then
    log_val "Skipping in-container GPU check (MODEL_HOST is not local)."
    return 0
  fi
  if docker compose -f "$COMPOSE_FILE" --project-directory "$(dirname "$COMPOSE_FILE")" exec -T ollama nvidia-smi -L >/dev/null 2>&1; then
    log_val "GPU visible inside ollama container (nvidia-smi -L OK)."
  else
    log_val "GPU check inside ollama skipped or failed (install NVIDIA Container Toolkit + drivers, or Ollama falls back to CPU)."
  fi
  return 0
}
