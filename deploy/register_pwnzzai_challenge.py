#!/usr/bin/env python3
"""
Create the PwnzzAI docker challenge and an optional static flag via CTFd API v1.

Requires an admin API token (Admin Panel → Settings → API Tokens).

Environment:
  CTFD_URL          Base URL, e.g. http://127.0.0.1:8000
  CTFD_API_TOKEN    Admin API token (alias: CTFD_API_KEY)
  PWNZZAI_IMAGE     Image tag participants spawn (default: pwnzzai-workshop:latest)
  CHALLENGE_NAME    (default: PwnzzAI Workshop)
  CHALLENGE_FLAG    If set, creates a static flag for the challenge

  DOCKER_CHALLENGES_PUBLIC_HOST  Required in repo-root .env — public hostname/IP for challenge links
  PWNZZAI_PUBLIC_HOST            Alias for the above (optional)
  ALLOW_CHALLENGE_REGISTER_WITHOUT_PUBLIC_HOST=1  Only for unusual setups (e.g. remote CTFd API only)

Docker API (workshop stack on this host):
  Before creating the challenge, updates SQLite in the ``ctfd`` container so Docker hostname is
  ``docker-socket-proxy:2375``. Requires ``docker compose`` and a running ``ctfd`` service.
  CTFD_SKIP_DOCKER_CONFIG=1  Skip (configure Admin → Docker Config yourself)
  CTFD_DOCKER_API_HOST       Override hostname:port (default docker-socket-proxy:2375)
  CTFD_COMPOSE_FILE          Path to compose file relative to repo root (default deploy/docker-compose.workshop.yml)

Usage:
  CTFD_URL=http://127.0.0.1:8000 CTFD_API_TOKEN=xxx python3 deploy/register_pwnzzai_challenge.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _merge_repo_dotenv() -> None:
    """Fill missing os.environ keys from repo-root .env (so python3 deploy/register_*.py works without export)."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def require_docker_challenges_public_host() -> int:
    """Refuse to register the workshop challenge unless a public host is configured."""
    skip = (os.environ.get("ALLOW_CHALLENGE_REGISTER_WITHOUT_PUBLIC_HOST") or "").strip().lower()
    if skip in ("1", "true", "yes"):
        return 0
    d = (os.environ.get("DOCKER_CHALLENGES_PUBLIC_HOST") or "").strip()
    if not d:
        p = (os.environ.get("PWNZZAI_PUBLIC_HOST") or "").strip()
        if p:
            os.environ["DOCKER_CHALLENGES_PUBLIC_HOST"] = p
            d = p
    if d:
        return 0
    print(
        "DOCKER_CHALLENGES_PUBLIC_HOST is not set. Add it to the repo-root .env file "
        "(public hostname or IP participants use in a browser — no http://). "
        "See .env.example and scripts/ctfd_setup/README.md. "
        "To override (rare): ALLOW_CHALLENGE_REGISTER_WITHOUT_PUBLIC_HOST=1",
        file=sys.stderr,
    )
    return 1


def apply_ctfd_docker_config() -> int:
    """
    Point the docker_challenges plugin at docker-socket-proxy:2375 by updating SQLite
    inside the running ``ctfd`` container (same stack as deploy/docker-compose.workshop.yml).

    Set CTFD_SKIP_DOCKER_CONFIG=1 to skip (e.g. remote CTFd or manual config).
    """
    skip = (os.environ.get("CTFD_SKIP_DOCKER_CONFIG") or "").strip().lower()
    if skip in ("1", "true", "yes"):
        return 0

    repo_root = Path(__file__).resolve().parent.parent
    compose_rel = os.environ.get("CTFD_COMPOSE_FILE", "deploy/docker-compose.workshop.yml")
    hostname = (os.environ.get("CTFD_DOCKER_API_HOST") or "docker-socket-proxy:2375").strip()
    inject_path = Path(__file__).resolve().parent / "inject_ctfd_docker_config.py"
    if not inject_path.is_file():
        print(f"Missing {inject_path}", file=sys.stderr)
        return 1

    cmd = [
        "docker",
        "compose",
        "-f",
        compose_rel,
        "exec",
        "-T",
        "-e",
        f"CTFD_DOCKER_API_HOST={hostname}",
        "ctfd",
        "python3",
        "-",
    ]
    try:
        subprocess.run(
            cmd,
            cwd=repo_root,
            input=inject_path.read_bytes(),
            check=True,
            timeout=120,
        )
    except FileNotFoundError:
        print(
            "docker CLI not found; cannot apply CTFd Docker config automatically. "
            "Set CTFD_SKIP_DOCKER_CONFIG=1 and configure Admin → Docker Config manually, "
            "or install Docker and ensure the workshop stack is running.",
            file=sys.stderr,
        )
        return 1
    except subprocess.CalledProcessError as e:
        print(
            "Could not inject docker_config into the CTFd container. "
            "Is the workshop stack up? Try: cd deploy && docker compose -f docker-compose.workshop.yml ps\n"
            "Set CTFD_SKIP_DOCKER_CONFIG=1 if CTFd runs elsewhere or you configure Docker manually.",
            file=sys.stderr,
        )
        return e.returncode
    except subprocess.TimeoutExpired:
        print("docker compose exec timed out.", file=sys.stderr)
        return 1
    return 0


def register_pwnzzai_challenge() -> int:
    _merge_repo_dotenv()

    base = os.environ.get("CTFD_URL", "http://127.0.0.1:8000").rstrip("/")
    token = (os.environ.get("CTFD_API_TOKEN") or os.environ.get("CTFD_API_KEY") or "").strip()
    image = os.environ.get("PWNZZAI_IMAGE", "pwnzzai-workshop:latest").strip()
    name = os.environ.get("CHALLENGE_NAME", "PwnzzAI Workshop").strip()
    flag = os.environ.get("CHALLENGE_FLAG", "").strip()

    if not token:
        print("CTFD_API_TOKEN is required.", file=sys.stderr)
        return 1

    if require_docker_challenges_public_host() != 0:
        return 1

    if apply_ctfd_docker_config() != 0:
        return 1

    headers = _headers(token)

    payload = {
        "name": name,
        "category": "Workshop",
        "description": (
            "Start your personal PwnzzAI instance with the button on this page. "
            "Use the shown host and port to open the app in your browser."
        ),
        "value": 1,
        "type": "docker",
        "state": "visible",
        "max_attempts": 0,
        "docker_image": image,
        "connection_info": "http://host:port (updated when your instance starts)",
    }

    req = urllib.request.Request(
        f"{base}/api/v1/challenges",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err = e.read().decode(errors="replace")
        print(f"HTTP {e.code} creating challenge: {err}", file=sys.stderr)
        return 1

    if not body.get("success"):
        print(json.dumps(body, indent=2), file=sys.stderr)
        return 1

    chal_id = body["data"]["id"]
    print(f"Created challenge id={chal_id} ({name}) docker_image={image}")

    if flag:
        fpayload = {
            "challenge_id": chal_id,
            "content": flag,
            "type": "static",
            "data": "",
        }
        freq = urllib.request.Request(
            f"{base}/api/v1/flags",
            data=json.dumps(fpayload).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(freq, timeout=60) as resp:
                fbody = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            err = e.read().decode(errors="replace")
            print(f"HTTP {e.code} creating flag: {err}", file=sys.stderr)
            return 1
        if not fbody.get("success"):
            print(json.dumps(fbody, indent=2), file=sys.stderr)
            return 1
        print("Flag created.")

    return 0


def main() -> int:
    return register_pwnzzai_challenge()


if __name__ == "__main__":
    raise SystemExit(main())
