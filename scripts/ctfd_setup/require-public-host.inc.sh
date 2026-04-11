# shellcheck shell=bash
# Source from bootstrap / redeploy after loading repo-root .env:
#   source "${SCRIPT_DIR}/require-public-host.inc.sh"
#   require_docker_challenges_public_host "${ENV_FILE}"

require_docker_challenges_public_host() {
  local env_hint="${1:-repo-root .env}"

  if [[ -z "${DOCKER_CHALLENGES_PUBLIC_HOST:-}" && -n "${PWNZZAI_PUBLIC_HOST:-}" ]]; then
    export DOCKER_CHALLENGES_PUBLIC_HOST="${PWNZZAI_PUBLIC_HOST}"
  fi

  local v="${DOCKER_CHALLENGES_PUBLIC_HOST:-}"
  v="${v#"${v%%[![:space:]]*}"}"
  v="${v%"${v##*[![:space:]]}"}"

  if [[ -z "$v" ]]; then
    printf '[%s] [ERROR] DOCKER_CHALLENGES_PUBLIC_HOST is not set.\n' "$(date -Is)" >&2
    printf 'Set it in %s to the public hostname or IP participants use in a browser (no http://).\n' "$env_hint" >&2
    printf 'Example: DOCKER_CHALLENGES_PUBLIC_HOST=ec2-203-0-113-1.compute.amazonaws.com\n' >&2
    printf 'Copy .env.example to .env if you do not have a file yet. See scripts/ctfd_setup/README.md\n' >&2
    exit 1
  fi

  export DOCKER_CHALLENGES_PUBLIC_HOST="$v"
}
