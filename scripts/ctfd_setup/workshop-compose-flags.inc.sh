# shellcheck shell=bash
# Shared by bootstrap / redeploy / teardown: which compose files to pass for the workshop stack.
#
# Prerequisites: PWNZZAI_ROOT set to the repository root (absolute path is fine).
# Optional (from .env if sourced by caller): PWNZZAI_OLLAMA_GPU
#   unset / empty — attach docker-compose.workshop.nvidia.yml only if nvidia-smi lists a GPU
#   0 — never attach (CPU-only Ollama; avoids CDI errors on cloud VMs without a GPU)
#   1 — always attach (host must have NVIDIA Container Toolkit + driver)
#
# Sets: PWNZZAI_WORKSHOP_COMPOSE_FLAGS — array of -f <file> flags (paths relative to deploy/).

pwnzzai_set_workshop_compose_flags() {
  PWNZZAI_WORKSHOP_COMPOSE_FLAGS=( -f docker-compose.workshop.yml )
  local nvidia="${PWNZZAI_ROOT}/deploy/docker-compose.workshop.nvidia.yml"
  if [[ ! -f "$nvidia" ]]; then
    return 0
  fi
  if [[ "${PWNZZAI_OLLAMA_GPU:-}" == "0" ]]; then
    return 0
  fi
  if [[ "${PWNZZAI_OLLAMA_GPU:-}" == "1" ]]; then
    PWNZZAI_WORKSHOP_COMPOSE_FLAGS+=( -f docker-compose.workshop.nvidia.yml )
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    PWNZZAI_WORKSHOP_COMPOSE_FLAGS+=( -f docker-compose.workshop.nvidia.yml )
  fi
}
