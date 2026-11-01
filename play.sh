#!/usr/bin/env bash
# Play fishnet in the browser (http://127.0.0.1:8000). Options: --port N, --depth N, --no-open
set -euo pipefail
cd "$(dirname "$0")"
exec uv run python app/server.py "$@"
