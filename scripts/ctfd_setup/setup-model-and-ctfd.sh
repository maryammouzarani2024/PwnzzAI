#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

log_info() { printf '[%s] [INFO] %s\n' "$(date -Is)" "$*"; }
log_error() { printf '[%s] [ERROR] %s\n' "$(date -Is)" "$*" >&2; }

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

# Default: start the local GPU Ollama service. Set COMPOSE_PROFILES= to omit it when using a remote model host.
export COMPOSE_PROFILES="${COMPOSE_PROFILES:-local-ollama}"

MODEL_TAG="${OLLAMA_MODEL:-mistral:7b}"
MODEL_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
COMPOSE_FILE="${ROOT_DIR}/deploy/docker-compose.workshop.yml"
DEPLOY_DIR="${ROOT_DIR}/deploy"

# shellcheck source=scripts/ctfd_setup/shared-model-validate.inc.sh
source "${ROOT_DIR}/scripts/ctfd_setup/shared-model-validate.inc.sh"

command -v docker >/dev/null 2>&1 || { log_error "docker is required"; exit 1; }
command -v curl >/dev/null 2>&1 || { log_error "curl is required"; exit 1; }

log_info "Checking for host port / model endpoint conflicts (shared Ollama must be the only consumer of the baked OLLAMA_HOST)"
check_port_11434_free_or_warn || exit 1

log_info "Bootstrapping CTFd + shared model infrastructure (one ollama container; all Pwnzai instances use OLLAMA_HOST → that service)"
"${ROOT_DIR}/scripts/ctfd_setup/bootstrap-ctfd-workshop.sh"

log_info "Verifying compose runs exactly one shared ollama service (not per-challenge)"
verify_single_ollama_service || exit 1
verify_gpu_in_ollama_optional

log_info "Ensuring shared model service is reachable (${MODEL_HOST})"
for _ in $(seq 1 30); do
  if curl -fsS "${MODEL_HOST}/api/tags" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
curl -fsS "${MODEL_HOST}/api/tags" >/dev/null

if [[ "${MODEL_HOST}" == "http://127.0.0.1:11434" || "${MODEL_HOST}" == "http://localhost:11434" || "${MODEL_HOST}" == "http://0.0.0.0:11434" ]]; then
  log_info "Pulling requested model into shared container: ${MODEL_TAG}"
  docker compose -f "$COMPOSE_FILE" --project-directory "$DEPLOY_DIR" exec -T ollama ollama pull "${MODEL_TAG}"
fi

log_info "Validating CTFd HTTP availability"
curl -fsS "http://127.0.0.1:8000/" >/dev/null || curl -fsS "http://127.0.0.1:8000/setup" >/dev/null

log_info "setup-model-and-ctfd completed"
