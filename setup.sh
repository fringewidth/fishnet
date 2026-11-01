#!/usr/bin/env bash
# Set up fishnet on an Apple Silicon Mac: the Qwenputer circuit IR, the Python environment,
# Qwen3.5-4B (about 9 GB), and a self-test of the chess layers. Then: ./play.sh
set -euo pipefail
cd "$(dirname "$0")"

QWENPUTER_REPO=https://github.com/divagr18/qwenputer.git
QWENPUTER_SHA=bec1a0c9654ac8151b6f36ed0154a19c53141384

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "note: this script targets Apple Silicon Macs; continuing anyway (Qwen will run on the CPU if MPS is missing)."
fi
mem_gb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
if (( mem_gb > 0 && mem_gb < 16 )); then
    echo "warning: ${mem_gb} GB of memory; fishnet needs about 12 GB free (Qwen3.5-4B in bf16 plus the chess layers)."
fi
if ! command -v uv >/dev/null; then
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "== 1/4 Qwenputer's circuit IR (MIT), pinned to ${QWENPUTER_SHA:0:7}"
if [[ ! -d vendor/qwenputer/.git ]]; then
    mkdir -p vendor/qwenputer
    git -C vendor/qwenputer init -q
    git -C vendor/qwenputer remote add origin "$QWENPUTER_REPO"
fi
if [[ "$(git -C vendor/qwenputer rev-parse HEAD 2>/dev/null || true)" != "$QWENPUTER_SHA" ]]; then
    git -C vendor/qwenputer fetch -q --depth 1 origin "$QWENPUTER_SHA"
    git -C vendor/qwenputer checkout -q FETCH_HEAD
fi

echo "== 2/4 Python environment"
uv sync

echo "== 3/4 Qwen3.5-4B (Hugging Face, about 9 GB; skipped if cached)"
uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-4B'))"

echo "== 4/4 self-test: the chess layers against the reference search, token by token"
uv run python circuit/selftest.py

echo
echo "Done. Start a game with: ./play.sh"
