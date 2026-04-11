#!/usr/bin/env bash
# Stop and remove the workshop stack (CTFd, Ollama, docker-socket-proxy).
#
# Does NOT remove the PwnzzAI workshop image (pwnzzai-workshop:latest) unless you pass --rmi-workshop.
# Participant-spawned containers (from the docker challenge) are on the host Docker daemon — see README.
#
# Usage (from repo root):
#   ./scripts/ctfd_setup/teardown-ctfd-workshop.sh
#   ./scripts/ctfd_setup/teardown-ctfd-workshop.sh --volumes   # also delete CTFd DB + Ollama model data
#   ./scripts/ctfd_setup/teardown-ctfd-workshop.sh --rmi-workshop
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PWNZZAI_ROOT="${PWNZZAI_ROOT:-$ROOT_DIR}"
ENV_FILE="${PWNZZAI_ROOT}/.env"
COMPOSE_FILE="${PWNZZAI_ROOT}/deploy/docker-compose.workshop.yml"

WITH_VOLUMES=0
RMI_WORKSHOP=0
for arg in "$@"; do
  case "$arg" in
    --volumes|-v) WITH_VOLUMES=1 ;;
    --rmi-workshop) RMI_WORKSHOP=1 ;;
    -h|--help)
      grep '^#' "$0" | head -20 | sed 's/^# \{0,1\}//'
      exit 0
      ;;
  esac
done

log_info() { printf '[%s] [INFO] %s\n' "$(date -Is)" "$*"; }
log_error() { printf '[%s] [ERROR] %s\n' "$(date -Is)" "$*" >&2; }

if [[ ! -f "$COMPOSE_FILE" ]]; then
  log_error "Cannot find ${COMPOSE_FILE}"
  exit 1
fi

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

# Compose still interpolates ctfd.environment; use a placeholder if .env is gone or incomplete.
if [[ -z "${DOCKER_CHALLENGES_PUBLIC_HOST:-}" && -n "${PWNZZAI_PUBLIC_HOST:-}" ]]; then
  export DOCKER_CHALLENGES_PUBLIC_HOST="${PWNZZAI_PUBLIC_HOST}"
fi
if [[ -z "${DOCKER_CHALLENGES_PUBLIC_HOST:-}" ]]; then
  export DOCKER_CHALLENGES_PUBLIC_HOST="127.0.0.1"
  log_info "DOCKER_CHALLENGES_PUBLIC_HOST not set — using a placeholder so compose can run (teardown only)."
fi

if ! command -v docker >/dev/null 2>&1; then
  log_error "Docker is not installed or not in your PATH."
  exit 1
fi

if docker info >/dev/null 2>&1; then
  dc() { docker compose "$@"; }
elif command -v sudo >/dev/null 2>&1 && sudo docker info >/dev/null 2>&1; then
  log_info "Using sudo for Docker commands."
  dc() { sudo docker compose "$@"; }
else
  log_error "Cannot talk to Docker."
  exit 1
fi

DOWN_ARGS=(down --remove-orphans)
if [[ "$WITH_VOLUMES" -eq 1 ]]; then
  DOWN_ARGS+=(--volumes)
  log_info "Will remove compose volumes (CTFd uploads/database and Ollama data)."
else
  log_info "Keeping compose volumes (add --volumes to delete CTFd + Ollama data)."
fi

log_info "Stopping workshop stack..."
cd "${PWNZZAI_ROOT}/deploy"
CTFD_SECRET_KEY="${CTFD_SECRET_KEY:-}" DOCKER_CHALLENGES_PUBLIC_HOST="${DOCKER_CHALLENGES_PUBLIC_HOST}" \
  dc -f docker-compose.workshop.yml "${DOWN_ARGS[@]}"

if [[ "$RMI_WORKSHOP" -eq 1 ]]; then
  WORKSHOP_TAG="${PWNZZAI_WORKSHOP_IMAGE:-pwnzzai-workshop:latest}"
  log_info "Removing workshop image ${WORKSHOP_TAG} (if present)..."
  if docker info >/dev/null 2>&1; then
    docker rmi "$WORKSHOP_TAG" 2>/dev/null || log_info "Image ${WORKSHOP_TAG} was not present (ok)."
  else
    sudo docker rmi "$WORKSHOP_TAG" 2>/dev/null || log_info "Image ${WORKSHOP_TAG} was not present (ok)."
  fi
fi

log_info "—"
log_info "Compose stack is down."
log_info "Spawned participant containers (if any) may still exist on this host — run: docker ps -a"
log_info "—"
