#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/home/users/mingxiao.li/git/quantization/quantx}"
PYTHON_BIN="${PYTHON_BIN:-/home/users/mingxiao.li/anaconda3/envs/test/bin/python}"
LOG_FILE="${LOG_FILE:-/tmp/quantx_server.log}"

cd "$ROOT"

PIDS=$(ps -ef | awk '/quantx\.server\.app/ && !/awk/ {print $2}')
if [[ -n "${PIDS}" ]]; then
  kill ${PIDS} || true
  sleep 1
fi

setsid "$PYTHON_BIN" -m quantx.server.app > "$LOG_FILE" 2>&1 < /dev/null &
PID=$!
sleep 2

cat <<EOF
{
  "ok": true,
  "pid": ${PID},
  "url": "http://127.0.0.1:8000",
  "log_file": "${LOG_FILE}"
}
EOF
