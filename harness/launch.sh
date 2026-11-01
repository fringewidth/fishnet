#!/bin/bash
# launch.sh OPPONENT(opus|astra) NODES... : one game per colour per budget, in the background
# RESUME=1 launch.sh ... continues each game from its runs/ log instead of starting fresh
cd "$(dirname "$0")"
opp=$1; shift
mkdir -p runs
for n in "$@"; do for c in white black; do
  t="${opp}-n${n}-${c}"
  if [ -n "${RESUME:-}" ]; then
    grep -E "^$t ply " "runs/$t.log" > "runs/$t.moves" && mv "runs/$t.moves" "runs/$t.log"
    nohup uv run -q --no-project --with chess --with 'mcp<2' python match.py "$n" "$c" "$t" "$opp" --resume >> "runs/$t.log" 2>&1 &
  else
    rm -rf "logs/$t"
    nohup uv run -q --no-project --with chess --with 'mcp<2' python match.py "$n" "$c" "$t" "$opp" > "runs/$t.log" 2>&1 &
  fi
done; done
