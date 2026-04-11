#!/usr/bin/env python3
"""
Run inside the CTFd container (stdin: ``python3 -``). Updates ``docker_config`` so the
Docker Challenges plugin uses the compose Docker API proxy.

Environment:
  CTFD_DOCKER_API_HOST  Hostname:port (default: docker-socket-proxy:2375)
"""
from __future__ import annotations

import os
import sqlite3
import sys

DB = "/var/uploads/ctfd.db"
DEFAULT_HOST = "docker-socket-proxy:2375"


def main() -> int:
    host = (os.environ.get("CTFD_DOCKER_API_HOST") or DEFAULT_HOST).strip()
    if len(host) > 64:
        print("CTFD_DOCKER_API_HOST must be at most 64 characters.", file=sys.stderr)
        return 1
    conn = sqlite3.connect(DB)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM docker_config WHERE id = 1")
        if cur.fetchone():
            cur.execute(
                """
                UPDATE docker_config
                SET hostname = ?, tls_enabled = 0,
                    ca_cert = NULL, client_cert = NULL, client_key = NULL,
                    repositories = NULL
                WHERE id = 1
                """,
                (host,),
            )
        else:
            cur.execute(
                """
                INSERT INTO docker_config
                    (id, hostname, tls_enabled, ca_cert, client_cert, client_key, repositories)
                VALUES (1, ?, 0, NULL, NULL, NULL, NULL)
                """,
                (host,),
            )
        conn.commit()
    finally:
        conn.close()
    print(f"CTFd docker_config: hostname={host!r}, tls_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
