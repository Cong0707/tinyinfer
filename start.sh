#!/usr/bin/env bash
# start.sh - start the resident tinyinfer engine with its OpenAI-compatible API.
#
# The engine loads once and picks its runtime path from the checkpoint itself, so whoever
# starts it decides which model is served. Every client then just connects.
#
#   ./start.sh --model /models/DeepSeek-V4.1-Flash-1017              # foreground
#   ./start.sh --model DIR --bg                                      # background
#   TINYINFER_MODEL=DIR ./start.sh                                   # via env
#   ./start.sh --model '/diskA/model;/diskB/model'                   # striped over disks
#
# Environment overrides: TINYINFER_MODEL, TINYINFER_TIER, TINYINFER_HOST, TINYINFER_PORT.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$here/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL="${TINYINFER_MODEL:-}"
TIER="${TINYINFER_TIER:-aggressive}"
HOST_ADDR="${TINYINFER_HOST:-127.0.0.1}"
PORT="${TINYINFER_PORT:-8000}"
BG=0

usage() {
  sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="${2:-}"; shift 2;;
    --tier) TIER="${2:-}"; shift 2;;
    --host) HOST_ADDR="${2:-}"; shift 2;;
    --port) PORT="${2:-}"; shift 2;;
    --bg) BG=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done

if [ -z "$MODEL" ]; then
  echo 'no checkpoint given: pass --model <dir>, or set TINYINFER_MODEL' >&2
  exit 1
fi
# a striped checkpoint is a list separated by ';' or ','; check the first entry
first="${MODEL%%[;,]*}"
if [ ! -e "$first" ]; then
  echo "checkpoint not found: $first" >&2
  exit 1
fi

PY="$here/.venv/bin/python"
[ -x "$PY" ] || PY="python"
if [ ! -x "$PY" ] && ! command -v "$PY" >/dev/null 2>&1; then
  echo 'no python found: run ./scripts/bootstrap.sh first' >&2
  exit 1
fi

echo "[tinyinfer] starting resident server on http://${HOST_ADDR}:${PORT} (tier=${TIER})"
if [ "$BG" = "1" ]; then
  nohup "$PY" -m tinyinfer.server --backend tiny --model-dir "$MODEL" \
        --tier "$TIER" --host "$HOST_ADDR" --port "$PORT" \
        > "$here/serve.log" 2> "$here/serve.err.log" &
  echo "[tinyinfer] background pid $!; tail serve.log for readiness"
else
  exec "$PY" -m tinyinfer.server --backend tiny --model-dir "$MODEL" \
       --tier "$TIER" --host "$HOST_ADDR" --port "$PORT"
fi
