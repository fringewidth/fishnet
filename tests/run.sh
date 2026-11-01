#!/usr/bin/env bash
# The verification suite, quickest first. ./tests/run.sh [--full] (--full adds depth-3 checks).
set -euo pipefail
cd "$(dirname "$0")/.."
run() { echo "== $*"; uv run python "$@"; }
run tests/test_ref.py 200          # move generation vs python-chess; token machine vs recursive alpha-beta
run tests/test_ref2.py 20 3        # the circuit's state layout vs the verified search
run tests/selftest.py              # the layers generate whole searches alone, == the reference
run tests/test_bridge.py 2         # Qwen's text -> FEN parse -> search -> printed move, exact IR
run tests/test_sparse.py 2         # the Mac executor vs the IR on every value
run tests/check_all.py 2           # IR next tokens, fp32 spans, executor
if [[ "${1:-}" == "--full" ]]; then
    run tests/test_circuit.py 3 5  # every stage's values vs the spec, depth 3
    run tests/broad_test.py        # mates, stalemate, promotion, random middlegames
fi
