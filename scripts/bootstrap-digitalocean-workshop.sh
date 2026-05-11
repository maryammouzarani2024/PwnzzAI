#!/usr/bin/env bash
# Bootstrap PwnzzAI + CTFd workshop on a DigitalOcean droplet (or similar) with extra host hardening.
#
# Intended to run ONCE as root (cloud-init / SSH session). Not idempotent for every step — review before re-run.
#
# Prerequisites:
#   • Ubuntu 22.04+ (apt). Other distros are not handled by this script.
#   • This repository at PWNZZAI_ROOT (default: parent of scripts/), OR set PWNZZAI_GIT_URL to clone into PWNZZAI_ROOT.
#   • Repo-root .env with DOCKER_CHALLENGES_PUBLIC_HOST, or DO_AUTO_PUBLIC_HOST=1 (uses DO metadata for public IPv4).
#
# Security notes (read before production):
#   • This host runs arbitrary code via the CTFd Docker Challenges plugin — isolate it (separate droplet, no shared creds).
#   • Use DigitalOcean Cloud Firewall: allow SSH from admin IPs only; allow 8000 (or 443 behind TLS) and TCP 30000–59999
#     for challenge instances; block Ollama (11434) from the internet (this script adds DOCKER-USER drops; still use DO FW).
#   • Put HTTPS in front of CTFd for real workshops (Caddy/nginx + Let’s Encrypt); this script does not configure TLS.
#
# Usage:
#   sudo ./scripts/bootstrap-digitalocean-workshop.sh
#
# Environment (optional):
#   PWNZZAI_ROOT=/opt/PwnzzAI
#   PWNZZAI_GIT_URL=https://github.com/you/PwnzzAI.git   # clone if PWNZZAI_ROOT is missing
#   DO_AUTO_PUBLIC_HOST=1                               # set DOCKER_CHALLENGES_PUBLIC_HOST from DO metadata
#   DO_AUTO_CTFD_SECRET=1                               # append CTFD_SECRET_KEY to .env if unset (openssl rand)
#   PWNZZAI_SKIP_HOST_HARDENING=1                         # only run workshop bootstrap (not recommended)
#   PWNZZAI_HARDEN_SSH=1                                # default: tighten sshd if root has authorized_keys
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PWNZZAI_ROOT="${PWNZZAI_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ENV_FILE="${PWNZZAI_ROOT}/.env"
ENV_EXAMPLE="${PWNZZAI_ROOT}/.env.example"

log_info() { printf '[%s] [INFO] %s\n' "$(date -Is)" "$*"; }
log_warn() { printf '[%s] [WARN] %s\n' "$(date -Is)" "$*" >&2; }
log_error() { printf '[%s] [ERROR] %s\n' "$(date -Is)" "$*" >&2; }

require_root() {
  if [[ "$(id -u)" -ne 0 ]]; then
    log_error "Run as root (e.g. sudo $0)."
    exit 1
  fi
}

require_apt() {
  if ! command -v apt-get >/dev/null 2>&1; then
    log_error "This script expects apt-get (Debian/Ubuntu)."
    exit 1
  fi
}

ensure_repo() {
  if [[ -f "${PWNZZAI_ROOT}/deploy/docker-compose.workshop.yml" ]]; then
    log_info "Using existing repo at ${PWNZZAI_ROOT}"
    return 0
  fi
  if [[ -n "${PWNZZAI_GIT_URL:-}" ]]; then
    log_info "Cloning ${PWNZZAI_GIT_URL} -> ${PWNZZAI_ROOT}"
    mkdir -p "$(dirname "$PWNZZAI_ROOT")"
    git clone "$PWNZZAI_GIT_URL" "$PWNZZAI_ROOT"
    return 0
  fi
  log_error "Repository not found at ${PWNZZAI_ROOT} and PWNZZAI_GIT_URL is not set."
  log_error "Clone PwnzzAI to that path or set PWNZZAI_GIT_URL."
  exit 1
}

detect_do_public_ipv4() {
  # DigitalOcean legacy metadata (no token required for this path on many droplets).
  curl -fsS --connect-timeout 2 --max-time 5 \
    http://169.254.169.254/metadata/v1/interfaces/public/0/ipv4/address 2>/dev/null || true
}

ensure_env_file() {
  if [[ -f "$ENV_FILE" ]]; then
    return 0
  fi
  if [[ -f "$ENV_EXAMPLE" ]]; then
    log_warn "Creating ${ENV_FILE} from .env.example — edit secrets before sharing the host."
    cp -a "$ENV_EXAMPLE" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    return 0
  fi
  log_error "Missing ${ENV_FILE} and ${ENV_EXAMPLE}. Cannot continue."
  exit 1
}

maybe_auto_public_host() {
  if [[ "${DO_AUTO_PUBLIC_HOST:-}" != "1" ]]; then
    return 0
  fi
  if grep -q '^[[:space:]]*DOCKER_CHALLENGES_PUBLIC_HOST=[^[:space:]]' "$ENV_FILE" 2>/dev/null; then
    log_info "DOCKER_CHALLENGES_PUBLIC_HOST already set in .env"
    return 0
  fi
  local ip
  ip="$(detect_do_public_ipv4)"
  if [[ -z "$ip" ]]; then
    log_error "DO_AUTO_PUBLIC_HOST=1 but could not read public IPv4 from metadata. Set DOCKER_CHALLENGES_PUBLIC_HOST in .env."
    exit 1
  fi
  printf '\n# Added by bootstrap-digitalocean-workshop.sh (DO metadata)\nDOCKER_CHALLENGES_PUBLIC_HOST=%s\n' "$ip" >>"$ENV_FILE"
  log_info "Set DOCKER_CHALLENGES_PUBLIC_HOST=${ip} in .env"
}

maybe_auto_ctfd_secret() {
  if [[ "${DO_AUTO_CTFD_SECRET:-}" != "1" ]]; then
    return 0
  fi
  if grep -q '^[[:space:]]*CTFD_SECRET_KEY=[^[:space:]]' "$ENV_FILE" 2>/dev/null; then
    log_info "CTFD_SECRET_KEY already set in .env"
    return 0
  fi
  local secret
  secret="$(openssl rand -hex 32)"
  printf '\n# Added by bootstrap-digitalocean-workshop.sh\nCTFD_SECRET_KEY=%s\n' "$secret" >>"$ENV_FILE"
  chmod 600 "$ENV_FILE"
  log_info "Generated CTFD_SECRET_KEY and appended to .env (keep this file private)."
}

apt_install_git_curl() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y --no-install-recommends ca-certificates curl git openssl
}

apt_install() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get install -y --no-install-recommends \
    openssl iptables ufw fail2ban unattended-upgrades
}

configure_sysctl() {
  local f=/etc/sysctl.d/99-pwnzzai-workshop.conf
  if [[ -f "$f" ]]; then
    log_info "Sysctl already present: $f"
    return 0
  fi
  cat <<'SYSCTL' >"$f"
# PwnzzAI workshop — modest hardening (does not replace firewall discipline)
net.ipv4.conf.default.rp_filter = 1
net.ipv4.conf.all.rp_filter = 1
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.tcp_syncookies = 1
SYSCTL
  sysctl --system >/dev/null || true
  log_info "Wrote $f"
}

configure_unattended_upgrades() {
  if [[ -f /usr/bin/unattended-upgrade ]]; then
    echo 'Unattended-Upgrade::Automatic-Reboot "false";' >/etc/apt/apt.conf.d/99pwnzzai-auto
    systemctl enable unattended-upgrades 2>/dev/null || true
    log_info "unattended-upgrades enabled (security updates; no auto-reboot)."
  fi
}

configure_fail2ban() {
  systemctl enable fail2ban 2>/dev/null || true
  systemctl restart fail2ban 2>/dev/null || true
  log_info "fail2ban installed (default sshd jail where packaged)."
}

configure_ssh_hardening() {
  if [[ "${PWNZZAI_HARDEN_SSH:-1}" != "1" ]]; then
    return 0
  fi
  local ak=/root/.ssh/authorized_keys
  if [[ ! -s "$ak" ]]; then
    log_warn "Skipping aggressive SSH hardening: $ak missing or empty (avoid lockout)."
    return 0
  fi
  chmod 700 /root/.ssh 2>/dev/null || true
  chmod 600 "$ak" 2>/dev/null || true
  local drop=/etc/ssh/sshd_config.d/99-pwnzzai-workshop.conf
  cat <<'SSHD' >"$drop"
# PwnzzAI workshop — key-based access only when authorized_keys exists (installed by bootstrap-digitalocean-workshop.sh)
PasswordAuthentication no
KbdInteractiveAuthentication no
ChallengeResponseAuthentication no
PermitRootLogin prohibit-password
X11Forwarding no
MaxAuthTries 4
LoginGraceTime 30
ClientAliveInterval 300
ClientAliveCountMax 2
SSHD
  if sshd -t 2>/dev/null; then
    systemctl reload sshd 2>/dev/null || systemctl reload ssh 2>/dev/null || true
    log_info "SSH hardened (${drop}). Verify you can open a second session before closing this one."
  else
    log_warn "sshd -t failed; removed ${drop}"
    rm -f "$drop"
  fi
}

configure_ufw() {
  ufw default deny incoming
  ufw default allow outgoing
  ufw allow OpenSSH
  ufw allow 8000/tcp comment 'CTFd'
  ufw allow 30000:59999/tcp comment 'CTFd docker challenges'
  # Ollama must not be exposed publicly; rely on DOCKER-USER + Cloud FW. Do not add ufw allow 11434.
  ufw --force enable
  log_info "UFW enabled (SSH, 8000, 30000–59999). Docker may add iptables rules — use DOCKER-USER + Cloud Firewall."
}

install_docker_harden_systemd() {
  local sbin=/usr/local/sbin/pwnzzai-docker-ollama-isolate.sh
  cat <<'HARDEN' >"$sbin"
#!/usr/bin/env bash
# Restrict who can reach host-published Ollama (11434): allow Docker bridge, drop other interfaces.
set -euo pipefail
OLLAMA_PORT="${OLLAMA_ISOLATE_PORT:-11434}"
if ! iptables -S DOCKER-USER >/dev/null 2>&1; then
  exit 0
fi
if iptables-save | grep -q 'pwnzzai-ollama-isolate'; then
  exit 0
fi
# Order matters: permit Docker bridge sources first, then drop everyone else for this published port.
iptables -I DOCKER-USER 1 -p tcp --dport "$OLLAMA_PORT" -s 172.17.0.0/16 -j RETURN -m comment --comment 'pwnzzai-ollama-isolate'
iptables -I DOCKER-USER 2 -p tcp --dport "$OLLAMA_PORT" -j DROP -m comment --comment 'pwnzzai-ollama-isolate'
if ip6tables -S DOCKER-USER >/dev/null 2>&1; then
  if ! ip6tables-save | grep -q 'pwnzzai-ollama-isolate'; then
    ip6tables -I DOCKER-USER 1 -p tcp --dport "$OLLAMA_PORT" -j DROP -m comment --comment 'pwnzzai-ollama-isolate'
  fi
fi
HARDEN
  chmod 700 "$sbin"

  cat <<'UNIT' >/etc/systemd/system/pwnzzai-docker-ollama-isolate.service
[Unit]
Description=Isolate Docker-published Ollama port from non-bridge clients (PwnzzAI workshop)
After=docker.service
PartOf=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/pwnzzai-docker-ollama-isolate.sh

[Install]
WantedBy=multi-user.target
UNIT
  systemctl daemon-reload
  systemctl enable pwnzzai-docker-ollama-isolate.service
  log_info "Installed systemd unit pwnzzai-docker-ollama-isolate.service for port 11434."
}

run_workshop_bootstrap() {
  local bs="${PWNZZAI_ROOT}/scripts/ctfd_setup/bootstrap-ctfd-workshop.sh"
  if [[ ! -x "$bs" ]]; then
    chmod +x "$bs"
  fi
  log_info "Running workshop bootstrap: $bs"
  (cd "$PWNZZAI_ROOT" && exec "$bs")
}

host_hardening() {
  log_info "Installing packages and configuring host hardening..."
  apt_install
  configure_sysctl
  configure_unattended_upgrades
  configure_ufw
  configure_fail2ban
  configure_ssh_hardening
}

main() {
  require_root
  require_apt
  log_info "PwnzzAI DigitalOcean workshop bootstrap (repo: ${PWNZZAI_ROOT})"

  apt_install_git_curl
  ensure_repo
  ensure_env_file
  maybe_auto_public_host
  maybe_auto_ctfd_secret

  if [[ "${PWNZZAI_SKIP_HOST_HARDENING:-}" == "1" ]]; then
    log_warn "PWNZZAI_SKIP_HOST_HARDENING=1 — skipping UFW/fail2ban/sysctl/SSH."
  else
    host_hardening
  fi

  if ! command -v docker >/dev/null 2>&1; then
    log_info "Installing Docker Engine via get.docker.com ..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable --now docker
  else
    log_info "Docker already installed."
  fi

  install_docker_harden_systemd
  systemctl start pwnzzai-docker-ollama-isolate.service || true

  run_workshop_bootstrap

  log_info "—"
  log_info "Bootstrap steps on this host are complete."
  log_info "Next: (1) DigitalOcean Cloud Firewall — restrict SSH to staff; allow 8000 + 30000–59999; deny 11434 from world."
  log_info "       (2) Open http://<public>:8000 and finish the CTFd wizard."
  log_info "       (3) Admin → API token → ${PWNZZAI_ROOT}/scripts/ctfd_setup/register-pwnzzai-challenge.sh"
  log_info "       (4) Add HTTPS reverse proxy for production; this stack is HTTP-only by default."
  log_info "—"
}

main "$@"
