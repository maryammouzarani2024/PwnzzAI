#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_URL="${APP_URL:-http://127.0.0.1:8080}"
USERS="${USERS:-20}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-60}"

log_info() { printf '[%s] [INFO] %s\n' "$(date -Is)" "$*"; }

log_info "Running shared-model load test users=${USERS} timeout=${TIMEOUT_SECONDS}s target=${APP_URL}"

python3 - <<'PY'
import concurrent.futures
import json
import os
import sys
import time
from urllib.request import Request, urlopen

app_url = os.environ.get("APP_URL", "http://127.0.0.1:8080").rstrip("/")
users = int(os.environ.get("USERS", "20"))
timeout_s = int(os.environ.get("TIMEOUT_SECONDS", "60"))

payload = json.dumps({"message": "Reply with one short line about pizza safety."}).encode("utf-8")
url = f"{app_url}/chat-with-ollama-dos"

def hit(_i: int):
    start = time.time()
    req = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout_s) as resp:
        body = resp.read()
        json.loads(body.decode("utf-8"))
    elapsed = time.time() - start
    return elapsed

latencies = []
errors = []
start_all = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=users) as ex:
    futures = [ex.submit(hit, i) for i in range(users)]
    for fut in concurrent.futures.as_completed(futures):
        try:
            latencies.append(fut.result())
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

total = time.time() - start_all
latencies.sort()
successes = len(latencies)
if latencies:
    p95_idx = max(0, min(len(latencies) - 1, int(len(latencies) * 0.95) - 1))
    p95 = latencies[p95_idx]
    max_latency = latencies[-1]
else:
    p95 = float("inf")
    max_latency = float("inf")

print(f"successes={successes} errors={len(errors)} total_runtime_s={total:.2f} p95_s={p95:.2f} max_s={max_latency:.2f}")
if errors:
    print("sample_error=", errors[0], file=sys.stderr)

if successes < users or max_latency >= timeout_s:
    sys.exit(1)
PY

log_info "Load test passed."
