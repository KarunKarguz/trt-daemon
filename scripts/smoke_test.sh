#!/usr/bin/env bash
set -euo pipefail

ENGINE=${ENGINE_PATH:-models/mobilenetv2_fp32.plan}
SOCK=${SOCKET_PATH:-/tmp/trt-smoke.sock}
# Pass semicolon-delimited runtime shapes (e.g., "input=1,3,640,640;boxes=1,100,4") to exercise dynamic engines.
SMOKE_RUNTIME_SHAPES=${SMOKE_RUNTIME_SHAPES:-}

daemon_pid=""

resolve_path() {
  python3 - "$1" <<'PY'
import os
import sys

path = sys.argv[1]
print(os.path.realpath(path))
PY
}

run_native() {
  if [[ ! -f "${ENGINE}" ]]; then
    echo "[smoke] skipping handshake; engine not found at ${ENGINE}" >&2
    exit 0
  fi

  local -a daemon_cmd=("./build/trt-daemon" -e "${ENGINE}" -s "${SOCK}")
  if [[ -n "${SMOKE_RUNTIME_SHAPES:-}" ]]; then
    TRT_SHAPES="${SMOKE_RUNTIME_SHAPES}" "${daemon_cmd[@]}" &
  else
    "${daemon_cmd[@]}" &
  fi
  daemon_pid=$!

  cleanup() {
    local pid="${daemon_pid:-}"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" >/dev/null 2>&1 || true
      wait "${pid}" 2>/dev/null || true
    fi
    rm -f "${SOCK}"
  }
  trap cleanup EXIT

  for _ in {1..20}; do
    if [[ -n "${daemon_pid}" ]] && ! kill -0 "${daemon_pid}" 2>/dev/null; then
      echo "[smoke] daemon exited before socket became ready; likely missing GPU" >&2
      return 0
    fi
    [[ -S "${SOCK}" ]] && break
    sleep 0.5
  done

  if [[ ! -S "${SOCK}" ]]; then
    echo "[smoke] socket never appeared; skipping native smoke" >&2
    return 0
  fi

  python3 scripts/inspect_bindings.py --sock "${SOCK}"
}

wait_for_port() {
  local host=$1
  local port=$2
  python3 - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.5)
try:
    s.connect((host, port))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
}

port_available() {
  local host=$1
  local port=$2
  python3 - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind((host, port))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
}

find_free_port() {
  local host=$1
  python3 - "$host" <<'PY'
import socket
import sys

host = sys.argv[1]

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, 0))
port = s.getsockname()[1]
s.close()
print(port)
PY
}

run_docker() {
  if [[ ! -f "${ENGINE}" ]]; then
    echo "[smoke] skipping handshake; engine not found at ${ENGINE}" >&2
    exit 0
  fi

  if ! command -v docker >/dev/null 2>&1; then
    echo "[smoke] docker CLI not available; install docker or set SMOKE_MODE=native" >&2
    exit 1
  fi

  local engine_abs
  engine_abs=$(resolve_path "${ENGINE}")
  local engine_dir="$(dirname "${engine_abs}")"
  local engine_name="$(basename "${engine_abs}")"

  local image="${SMOKE_IMAGE:-trt-daemon:smoke}"
  local build_target="${SMOKE_BUILD_TARGET:-runtime}"
  local build_args=()

  if [[ -n "${SMOKE_BUILD_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    build_args=(${SMOKE_BUILD_ARGS})
  fi

  echo "[smoke] building Docker image '${image}' (target=${build_target})"
  docker build "${build_args[@]}" --target "${build_target}" -t "${image}" .

  local host_ip="${SMOKE_TCP_HOST:-127.0.0.1}"
  local host_port="${SMOKE_TCP_PORT:-18008}"
  local container_name="${SMOKE_CONTAINER_NAME:-trt-daemon-smoke-$$}"

  if [[ -n "${host_ip}" ]]; then
    if ! port_available "${host_ip}" "${host_port}"; then
      if [[ -n "${SMOKE_TCP_PORT:-}" ]]; then
        echo "[smoke] host port ${host_ip}:${host_port} already in use; override with SMOKE_TCP_PORT" >&2
        exit 1
      fi
      host_port=$(find_free_port "${host_ip}")
      echo "[smoke] host port in use; using alternate ${host_ip}:${host_port}" >&2
    fi
  fi

  local -a gpu_args=()
  local gpu_setting="${SMOKE_DOCKER_GPUS:-auto}"
  if [[ "${gpu_setting}" == "auto" ]]; then
    if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
      gpu_args=(--gpus all)
    fi
  elif [[ -n "${gpu_setting}" ]]; then
    gpu_args=(--gpus "${gpu_setting}")
  fi

  local -a volume_flags=(-v "${engine_dir}:/opt/host-engine:ro")
  local -a env_flags=(-e "ENGINE_PATH=/opt/host-engine/${engine_name}")

  if [[ -n "${ONNX_PATH:-}" ]]; then
    local onnx_abs
    onnx_abs=$(resolve_path "${ONNX_PATH}")
    local onnx_dir="$(dirname "${onnx_abs}")"
    local onnx_name="$(basename "${onnx_abs}")"
    volume_flags+=(-v "${onnx_dir}:/opt/host-onnx:ro")
    env_flags+=(-e "ONNX_PATH=/opt/host-onnx/${onnx_name}")
  fi

  if [[ -n "${TRT_OPTS:-}" ]]; then
    env_flags+=(-e "TRT_OPTS=${TRT_OPTS}")
  fi

  if [[ -n "${SMOKE_RUNTIME_SHAPES:-}" ]]; then
    env_flags+=(-e "TRT_SHAPES=${SMOKE_RUNTIME_SHAPES}")
  fi

  local publish_arg
  if [[ -n "${host_ip}" ]]; then
    publish_arg="${host_ip}:${host_port}:8008"
  else
    publish_arg="${host_port}:8008"
  fi

  echo "[smoke] starting runtime container '${container_name}'"
  local container_id
  container_id=$(docker run --rm --detach \
    --name "${container_name}" \
    -p "${publish_arg}" \
    "${gpu_args[@]}" \
    "${volume_flags[@]}" \
    "${env_flags[@]}" \
    "${image}")

  cleanup_container() {
    local cid="${container_id:-}"
    if [[ -n "${cid}" ]]; then
      docker stop "${cid}" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup_container EXIT

  local ready=0
  for _ in {1..40}; do
    if wait_for_port "${host_ip:-127.0.0.1}" "${host_port}"; then
      ready=1
      break
    fi
    sleep 0.5
  done

  if [[ "${ready}" != "1" ]]; then
    echo "[smoke] daemon did not become ready on ${host_ip:-127.0.0.1}:${host_port}" >&2
    docker logs "${container_id}" || true
    exit 1
  fi

  local handshake_attempts="${SMOKE_HANDSHAKE_ATTEMPTS:-5}"
  local handshake_delay="${SMOKE_HANDSHAKE_DELAY:-1}"
  local handshake_ok=0
  local handshake_output=""

  for ((attempt = 1; attempt <= handshake_attempts; attempt++)); do
    if handshake_output=$(python3 scripts/inspect_bindings.py --tcp "${host_ip:-127.0.0.1}" "${host_port}" 2>&1); then
      handshake_ok=1
      printf '%s\n' "${handshake_output}"
      break
    fi
    echo "[smoke] handshake attempt ${attempt}/${handshake_attempts} failed; retrying in ${handshake_delay}s" >&2
    echo "${handshake_output}" >&2
    if (( attempt < handshake_attempts )); then
      sleep "${handshake_delay}"
    fi
  done

  if [[ "${handshake_ok}" != "1" ]]; then
    echo "[smoke] handshake failed after ${handshake_attempts} attempts" >&2
    docker logs "${container_id}" || true
    exit 1
  fi

  cleanup_container
  trap - EXIT
}

main() {
  local mode="${SMOKE_MODE:-auto}"
  case "${mode}" in
    native)
      run_native
      ;;
    docker)
      run_docker
      ;;
    auto)
      if [[ -f /.dockerenv ]]; then
        run_native
      elif command -v docker >/dev/null 2>&1; then
        run_docker
      else
        run_native
      fi
      ;;
    *)
      echo "[smoke] unknown SMOKE_MODE '${mode}' (expected auto|native|docker)" >&2
      exit 1
      ;;
  esac
}

main "$@"
