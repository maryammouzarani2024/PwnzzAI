#!/usr/bin/env bash
# Step 2: Register the PwnzzAI docker challenge in CTFd via the admin API.
#
# Run this AFTER:
#   1) ./scripts/ctfd_setup/bootstrap-ctfd-workshop.sh
#   2) CTFd setup wizard (first-run admin account)
#   3) Admin API token: Admin → Settings → API Tokens → Generate
#
# Docker API: deploy/register_pwnzzai_challenge.py updates docker_config in the ctfd container
# (docker-socket-proxy:2375) via docker compose exec. Set CTFD_SKIP_DOCKER_CONFIG=1 to skip.
#
# Usage:
#   export CTFD_API_TOKEN='your-token'
#   ./scripts/ctfd_setup/register-pwnzzai-challenge.sh
#
# Or pass the token as the first argument (not recommended for shell history on shared machines):
#   ./scripts/ctfd_setup/register-pwnzzai-challenge.sh 'your-token'
#
# Required in repo-root .env (see .env.example):
#   DOCKER_CHALLENGES_PUBLIC_HOST — public hostname or IP for challenge links
# Optional: ALLOW_CHALLENGE_REGISTER_WITHOUT_PUBLIC_HOST=1 — only for unusual setups
#
# Optional env (see deploy/register_pwnzzai_challenge.py):
#   CTFD_URL              (default: http://127.0.0.1:8000)
#   PWNZZAI_IMAGE         (default: pwnzzai-workshop:latest or PWNZZAI_WORKSHOP_IMAGE)
#   CHALLENGE_NAME
#   CHALLENGE_FLAG
#   CTFD_SKIP_DOCKER_CONFIG  Set to 1 if CTFd is not the local compose ctfd service
#   CTFD_DOCKER_API_HOST     (default: docker-socket-proxy:2375)
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PWNZZAI_ROOT="${PWNZZAI_ROOT:-$ROOT_DIR}"
ENV_FILE="${PWNZZAI_ROOT}/.env"

log_info() { printf '[%s] [INFO] %s\n' "$(date -Is)" "$*"; }
log_error() { printf '[%s] [ERROR] %s\n' "$(date -Is)" "$*" >&2; }

if [[ -f "$ENV_FILE" ]]; then
  log_info "Loading ${ENV_FILE}"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

if [[ "${ALLOW_CHALLENGE_REGISTER_WITHOUT_PUBLIC_HOST:-}" != "1" ]]; then
  # shellcheck source=scripts/ctfd_setup/require-public-host.inc.sh
  source "${PWNZZAI_ROOT}/scripts/ctfd_setup/require-public-host.inc.sh"
  require_docker_challenges_public_host "${ENV_FILE}"
fi

TOKEN="${CTFD_API_TOKEN:-${CTFD_API_KEY:-}}"
if [[ -n "${1:-}" ]]; then
  TOKEN="$1"
fi
TOKEN="${TOKEN:-}"

if [[ -z "$TOKEN" ]]; then
  log_error "No API token found. Set CTFD_API_TOKEN (or CTFD_API_KEY) in your environment or .env,"
  log_error "or pass the token as the first argument to this script."
  echo ""
  echo "Example:"
  echo "  export CTFD_API_TOKEN='...'"
  echo "  $0"
  echo ""
  echo "See the header comments in this script for full options."
  exit 1
fi

export CTFD_URL="${CTFD_URL:-http://127.0.0.1:8000}"
export CTFD_API_TOKEN="$TOKEN"
WORKSHOP_TAG="${PWNZZAI_WORKSHOP_IMAGE:-pwnzzai-workshop:latest}"
export PWNZZAI_IMAGE="${PWNZZAI_IMAGE:-$WORKSHOP_TAG}"
export CHALLENGE_FLAG="${CHALLENGE_FLAG:-}"

log_info "CTFd URL: ${CTFD_URL}"
log_info "Registering docker challenge with image: ${PWNZZAI_IMAGE}"

exec python3 "${PWNZZAI_ROOT}/deploy/register_pwnzzai_challenge.py"
