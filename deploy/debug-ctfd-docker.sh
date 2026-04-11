#!/usr/bin/env bash
# Run from the repo host (with the workshop stack up). Checks run *inside* the CTFd
# container so they hit the same URL as the plugin: http://docker-socket-proxy:2375
#
# Usage:
#   cd deploy && ./debug-ctfd-docker.sh
#   IMAGE=pwnzzai-workshop:latest ./debug-ctfd-docker.sh
#
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export IMAGE="${IMAGE:-pwnzzai-workshop:latest}"

cd "$DEPLOY_DIR"
if ! docker compose -f docker-compose.workshop.yml ps --status running --quiet ctfd >/dev/null 2>&1; then
  echo "CTFd container is not running. Start: docker compose -f deploy/docker-compose.workshop.yml up -d" >&2
  exit 1
fi

docker compose -f docker-compose.workshop.yml exec -T ctfd python3 - <<'PY'
"""Mirror plugin: version + image inspect (encoded image name)."""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

IMAGE = os.environ.get("IMAGE", "pwnzzai-workshop:latest")
HOST = "http://docker-socket-proxy:2375"


def get(path: str) -> tuple[int, str]:
    req = urllib.request.Request(f"{HOST}{path}", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.getcode(), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


print("=== Docker version (proxy) ===")
code, body = get("/version")
print("HTTP", code, body[:400])
if code != 200:
    sys.exit(1)

print("\n=== Image inspect:", IMAGE, "===")
enc = urllib.parse.quote(IMAGE, safe="")
code, body = get(f"/images/{enc}/json")
print("HTTP", code)
if code != 200:
    print(body[:1200])
    print("\nImage missing on the Docker host. Build/tag the workshop image on this machine, then retry.")
    sys.exit(1)

meta = json.loads(body)
ports = (meta.get("Config") or {}).get("ExposedPorts") or {}
print("ExposedPorts:", list(ports.keys()) or "NONE (plugin cannot map ports)")
PY
