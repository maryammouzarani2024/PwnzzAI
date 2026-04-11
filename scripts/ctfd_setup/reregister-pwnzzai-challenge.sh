#!/usr/bin/env bash
# Delete the PwnzzAI challenge by name (if present), then create it again.
# Same environment variables as register-pwnzzai-challenge.sh
#
# Usage:
#   export CTFD_API_TOKEN='...'
#   ./scripts/ctfd_setup/reregister-pwnzzai-challenge.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PWNZZAI_ROOT="${PWNZZAI_ROOT:-$ROOT_DIR}"
ENV_FILE="${PWNZZAI_ROOT}/.env"

log_info() { printf '[%s] [INFO] %s\n' "$(date -Is)" "$*"; }

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
if [[ -z "${TOKEN:-}" ]]; then
  echo "Set CTFD_API_TOKEN or pass token as first argument." >&2
  exit 1
fi

export CTFD_URL="${CTFD_URL:-http://127.0.0.1:8000}"
export CTFD_API_TOKEN="$TOKEN"
WORKSHOP_TAG="${PWNZZAI_WORKSHOP_IMAGE:-pwnzzai-workshop:latest}"
export PWNZZAI_IMAGE="${PWNZZAI_IMAGE:-$WORKSHOP_TAG}"
export CHALLENGE_FLAG="${CHALLENGE_FLAG:-}"

exec python3 "${PWNZZAI_ROOT}/deploy/reregister_pwnzzai_challenge.py"
