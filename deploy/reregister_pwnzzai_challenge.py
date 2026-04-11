#!/usr/bin/env python3
"""
Remove existing PwnzzAI challenge(s) by name, then create a fresh one (same as register_pwnzzai_challenge.py).

Uses CTFd API v1: list challenges, DELETE by id, then POST create.

Environment (same as register_pwnzzai_challenge.py):
  CTFD_URL          Base URL, e.g. http://127.0.0.1:8000
  CTFD_API_TOKEN    Admin API token (alias: CTFD_API_KEY)
  CHALLENGE_NAME    Match challenges to delete (default: PwnzzAI Workshop)
  PWNZZAI_IMAGE, CHALLENGE_FLAG — passed through to registration

Usage:
  CTFD_URL=http://127.0.0.1:8000 CTFD_API_TOKEN=xxx python3 deploy/reregister_pwnzzai_challenge.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

# Allow `python3 deploy/reregister_pwnzzai_challenge.py` from repo root
_DEPLOY = os.path.dirname(os.path.abspath(__file__))
if _DEPLOY not in sys.path:
    sys.path.insert(0, _DEPLOY)

from register_pwnzzai_challenge import register_pwnzzai_challenge


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _get_challenge_list(base: str, token: str, query: str) -> dict:
    headers = _headers(token)
    url = f"{base}/api/v1/challenges{query}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def list_challenge_ids_by_name(base: str, token: str, name: str) -> list[int]:
    """Return challenge ids whose name equals `name` (admin token required)."""
    body = None
    for query in ("?view=admin", ""):
        try:
            body = _get_challenge_list(base, token, query)
            break
        except urllib.error.HTTPError as e:
            if e.code != 400 and e.code != 404:
                err = e.read().decode(errors="replace")
                print(f"HTTP {e.code} listing challenges: {err}", file=sys.stderr)
                raise
    if body is None:
        print("Could not list challenges.", file=sys.stderr)
        raise RuntimeError("list challenges failed")
    if not body.get("success"):
        print(json.dumps(body, indent=2), file=sys.stderr)
        raise RuntimeError("list challenges failed")
    ids: list[int] = []
    for row in body.get("data") or []:
        if row.get("name") == name:
            ids.append(int(row["id"]))
    return ids


def delete_challenge(base: str, token: str, challenge_id: int) -> None:
    headers = _headers(token)
    req = urllib.request.Request(
        f"{base}/api/v1/challenges/{challenge_id}",
        headers=headers,
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err = e.read().decode(errors="replace")
        print(f"HTTP {e.code} deleting challenge {challenge_id}: {err}", file=sys.stderr)
        raise
    if not body.get("success"):
        print(json.dumps(body, indent=2), file=sys.stderr)
        raise RuntimeError(f"delete challenge {challenge_id} failed")


def main() -> int:
    base = os.environ.get("CTFD_URL", "http://127.0.0.1:8000").rstrip("/")
    token = (os.environ.get("CTFD_API_TOKEN") or os.environ.get("CTFD_API_KEY") or "").strip()
    name = os.environ.get("CHALLENGE_NAME", "PwnzzAI Workshop").strip()

    if not token:
        print("CTFD_API_TOKEN is required.", file=sys.stderr)
        return 1

    try:
        ids = list_challenge_ids_by_name(base, token, name)
    except Exception:
        return 1

    if not ids:
        print(f"No challenge named {name!r} found; registering only.")
    for cid in ids:
        print(f"Deleting challenge id={cid} ({name})...")
        try:
            delete_challenge(base, token, cid)
        except Exception:
            return 1
        print(f"Deleted challenge id={cid}.")

    return register_pwnzzai_challenge()


if __name__ == "__main__":
    raise SystemExit(main())
