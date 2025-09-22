#!/usr/bin/env bash
set -euo pipefail

ENGINE=${ENGINE_PATH:-/workspace/models/mobilenetv2_fp32.plan}
SOCK=${SOCKET_PATH:-/tmp/trt-smoke.sock}

if [[ ! -f "${ENGINE}" ]]; then
  echo "[smoke] skipping handshake; engine not found at ${ENGINE}" >&2
  exit 0
fi

./build/trt-daemon -e "${ENGINE}" -s "${SOCK}" &
DAEMON_PID=$!

cleanup() {
  if kill -0 "${DAEMON_PID}" 2>/dev/null; then
    kill "${DAEMON_PID}" >/dev/null 2>&1 || true
    wait "${DAEMON_PID}" 2>/dev/null || true
  fi
  rm -f "${SOCK}"
}
trap cleanup EXIT

for _ in {1..20}; do
  [[ -S "${SOCK}" ]] && break
  sleep 0.5
done

python3 scripts/inspect_bindings.py --sock "${SOCK}"

cleanup
