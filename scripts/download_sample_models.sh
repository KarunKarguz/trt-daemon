#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: download_sample_models.sh [--model NAME]

Fetch open-source ONNX sample models so smoke tests and container images
have something to work with. Currently supported models:
  - squeezenet (default)

The files are saved under ./models/.
USAGE
}

model="squeezenet"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      model="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

mkdir -p models

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to fetch sample models" >&2
  exit 1
fi

case "${model}" in
  squeezenet|squeezenet1|squeezenet1.1)
    url="https://github.com/onnx/models/raw/main/vision/classification/squeezenet/model/squeezenet1.1-7.onnx"
    target="models/squeezenet1.1.onnx"
    ;;
  *)
    echo "Unsupported model: ${model}" >&2
    usage
    exit 1
    ;;
 esac

if [[ -f "${target}" ]]; then
  echo "[download-models] ${target} already present; skipping" >&2
  exit 0
fi

tmp_file="${target}.tmp"
rm -f "${tmp_file}"

echo "[download-models] Fetching ${model} ONNX from ${url}" >&2
curl -L --fail --show-error --output "${tmp_file}" "${url}"

mv "${tmp_file}" "${target}"

echo "[download-models] Saved ${target}" >&2
