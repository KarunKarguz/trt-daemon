#!/usr/bin/env bash
set -euo pipefail

ENGINE_PATH="${ENGINE_PATH:-/opt/models/mobilenetv2_fp32.plan}"
SOCKET_PATH="${SOCKET_PATH:-/run/trt.sock}"
EXPOSE_TCP="${EXPOSE_TCP:-1}"
TCP_PORT="${TCP_PORT:-8008}"

# Optional: auto-build engine from ONNX when ENGINE_PATH is missing
#   Provide ONNX_PATH and TRT_OPTS envs (e.g., "--memPoolSize=workspace:512 --fp16")
ONNX_PATH="${ONNX_PATH:-}"
TRT_OPTS="${TRT_OPTS:-}"

mkdir -p /run

if [[ ! -f "${ENGINE_PATH}" ]]; then
  if [[ -n "${ONNX_PATH}" && -f "${ONNX_PATH}" ]]; then
    echo "[entrypoint] ENGINE not found, building from ONNX..."
    # You can add shapes flags if your ONNX has dynamic dims; TRT often defaults to 1x3x224x224 for common CNNs.
    trtexec --onnx="${ONNX_PATH}" ${TRT_OPTS} --saveEngine="${ENGINE_PATH}"
  else
    echo "[entrypoint] ERROR: Engine not found at ${ENGINE_PATH} and no ONNX_PATH provided."
    ls -l /opt/models || true
    exit 1
  fi
fi

echo "[entrypoint] Starting TRT daemon with engine: ${ENGINE_PATH}  socket: ${SOCKET_PATH}"
# Start optional TCP proxy first (in background) so net clients can connect
if [[ "${EXPOSE_TCP}" == "1" ]]; then
  echo "[entrypoint] Launching TCP proxy on 0.0.0.0:${TCP_PORT} -> ${SOCKET_PATH}"
  socat TCP-LISTEN:${TCP_PORT},fork,reuseaddr UNIX-CONNECT:${SOCKET_PATH} &
fi

# Launch daemon in foreground (PID 1)
exec /app/trtdaemon/build/trt-daemon -e "${ENGINE_PATH}" -s "${SOCKET_PATH}"
