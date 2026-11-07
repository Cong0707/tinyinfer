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
# 对外服务的 API key（OpenAI 的 Authorization: Bearer <key>）；留空则不校验
API_KEY="${TINYINFER_API_KEY:-}"
# /v1/models 里暴露的名字
MODEL_ID="${TINYINFER_MODEL_ID:-}"
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
    --api-key) API_KEY="${2:-}"; shift 2;;
    --model-id) MODEL_ID="${2:-}"; shift 2;;
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

# 启动前确认显存没被别人占着。上一个实例没清理干净时，新进程会在
# cudaMalloc 上 OOM，而它是在模型加载到一半才炸的 —— 既浪费几分钟，
# 也把真正的错因埋在很深的调用栈里。这里提前拦下来。
if command -v nvidia-smi >/dev/null 2>&1; then
  _free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  _need="${TINYINFER_MIN_FREE_MB:-8000}"
  if [ -n "$_free" ] && [ "$_free" -lt "$_need" ]; then
    echo "[tinyinfer] only ${_free}MB of GPU memory free (need >= ${_need}MB)." >&2
    echo "           another process is probably still holding the card:" >&2
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/             /' >&2
    echo "           kill it (or wait for it to exit) before starting." >&2
    exit 4
  fi
fi

# 进程锁：一个 checkpoint 只能有一个常驻实例。重复启动会各自去抢显存，
# 第二个必然在 cudaMalloc 上 OOM，而第一个已经吃掉 36GB；更糟的是失败的
# 那个留下一个 Zl 僵尸进程，显存要等驱动回收才回来。
LOCK="$here/.serve.lock"
if [ -f "$LOCK" ]; then
  _old=$(cat "$LOCK" 2>/dev/null || echo "")
  if [ -n "$_old" ] && kill -0 "$_old" 2>/dev/null; then
    echo "[tinyinfer] already running (pid $_old); refusing to start a second one" >&2
    echo "           kill it first: kill $_old   (or rm $LOCK if the pid is stale)" >&2
    exit 3
  fi
  # stale lock (process died): reclaim it
  rm -f "$LOCK"
fi
printf '%s' "$$" > "$LOCK"
trap 'rm -f "$LOCK" 2>/dev/null' EXIT INT TERM
if [ ! -x "$PY" ] && ! command -v "$PY" >/dev/null 2>&1; then
  echo 'no python found: run ./scripts/bootstrap.sh first' >&2
  exit 1
fi

echo "[tinyinfer] starting resident server on http://${HOST_ADDR}:${PORT} (tier=${TIER})"
extra=()
[ -n "$API_KEY" ] && extra+=(--api-key "$API_KEY")
[ -n "$MODEL_ID" ] && extra+=(--model-id "$MODEL_ID")

if [ "$BG" = "1" ]; then
  nohup "$PY" -m tinyinfer.server --backend tiny --model-dir "$MODEL" \
        --tier "$TIER" --host "$HOST_ADDR" --port "$PORT" "${extra[@]}" \
        > "$here/serve.log" 2> "$here/serve.err.log" &
  echo "[tinyinfer] background pid $!; tail serve.log for readiness"
else
  exec "$PY" -m tinyinfer.server --backend tiny --model-dir "$MODEL" \
       --tier "$TIER" --host "$HOST_ADDR" --port "$PORT" "${extra[@]}"
fi
