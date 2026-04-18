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

APP_URL="${APP_URL:-http://127.0.0.1:8080}"
# probes use APP_BASE without trailing slash
APP_BASE="${APP_URL%/}"
CTFD_URL="${CTFD_URL:-http://127.0.0.1:8000}"

# shellcheck source=scripts/qa/probes-standalone.inc.sh
source "${ROOT_DIR}/scripts/qa/probes-standalone.inc.sh"

log_info "Gate 1/4: Run existing unit tests"
python3 -m pytest "${ROOT_DIR}/tests/unit"

log_info "Gate 2/4: Single PwnzzAI instance — full standalone model probes (same as docker-smoke-test.sh)"
run_standalone_model_probes

log_info "Gate 3/4: CTFd instance reachable for challenge wiring"
curl -fsS "${CTFD_URL}/" >/dev/null || curl -fsS "${CTFD_URL}/setup" >/dev/null

log_info "Gate 4/4: Mixed provider readiness (OpenAI session check + repeat model probes)"
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  log_info "Preconfigured OPENAI_API_KEY detected."
else
  log_info "No preconfigured OPENAI_API_KEY; OpenAI gate will rely on user session keys at runtime."
fi
curl -fsS "${APP_BASE}/check-openai-api-key" >/dev/null
run_standalone_model_probes

log_info "All intermediate gates passed."
