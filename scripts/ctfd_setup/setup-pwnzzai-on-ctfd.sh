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

if [[ -z "${CTFD_API_TOKEN:-${CTFD_API_KEY:-}}" ]]; then
  log_error "CTFD_API_TOKEN (or CTFD_API_KEY) is required to register the challenge."
  exit 1
fi

log_info "Registering/updating PwnzzAI challenge on CTFd"
"${ROOT_DIR}/scripts/ctfd_setup/register-pwnzzai-challenge.sh"

log_info "setup-pwnzzai-on-ctfd completed"
