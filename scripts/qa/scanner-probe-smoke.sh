#!/usr/bin/env bash
# Minimal HTTP smoke for scanner / CI inventory (no Garak/PyRIT bundled here).
set -euo pipefail
BASE="${APP_BASE:-http://127.0.0.1:5000}"
paths=(
  "/"
  "/basics"
  "/data-poisoning"
  "/direct-prompt-injection"
  "/indirect-prompt-injection"
  "/promotion-photo-claim"
  "/customer-support-safety"
  "/model-theft"
)
for p in "${paths[@]}"; do
  code=$(curl -sS -o /dev/null -w "%{http_code}" "${BASE}${p}")
  if [[ "$code" != "200" ]]; then
    echo "FAIL ${p} -> HTTP ${code}" >&2
    exit 1
  fi
  echo "OK ${p}"
done
echo "scanner-probe-smoke: all paths returned 200"
