"""One game: native Qwenputer engine at a node budget vs an LLM with board tools only.
    match.py NODES QWENPUTER_COLOR(white|black) TAG [opus|astra]
Writes games/TAG.pgn and per-move logs; prints one line per move and the result."""
import datetime
import os
import sys

import chess
import chess.pgn

from claude_player import ClaudePlayer
from codex_player import CodexPlayer
from players import Qwenputer, SearchMachine

spec, qcolor, tag = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs("games", exist_ok=True)
log_dir = f"logs/{tag}"
os.makedirs(log_dir, exist_ok=True)
q = SearchMachine(int(spec[1:])) if spec.startswith("d") else Qwenputer(int(spec))
opp = sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else "opus"
llm = (ClaudePlayer(model="claude-opus-5-5", effort="high", log_dir=log_dir) if opp == "opus"
       else CodexPlayer(model="gpt-6-astra", effort="high", log_dir=log_dir))
white, black = (q, llm) if qcolor == "white" else (llm, q)
b = chess.Board()
notes = []
# --resume: replay the SAN moves already logged in runs/TAG.log (e.g. after a usage-limit stop)
if "--resume" in sys.argv:
    import re
    for line in open(f"runs/{tag}.log"):
        mt = re.match(rf"{re.escape(tag)} ply +\d+ +(\S+) +(\S+) +(.*)$", line.rstrip("\n"))
        if mt:
            b.push_san(mt.group(2))
            notes.append(mt.group(3))
    print(f"{tag} resumed at ply {b.ply()}", flush=True)
while not b.is_game_over(claim_draw=True) and b.ply() < 300:
    p = white if b.turn == chess.WHITE else black
    m = p.move(b)
    assert m in b.legal_moves, (p.name, m.uci(), b.fen())
    san = b.san(m)
    notes.append(p.last)
    b.push(m)
    print(f"{tag} ply {b.ply():3d} {p.name:28s} {san:8s} {p.last}", flush=True)
result = b.result(claim_draw=True)
g = chess.pgn.Game.from_board(b)
g.headers.update({"Event": f"Qwenputer vs {llm.name} (native calibration)", "White": white.name, "Black": black.name,
                  "Result": result, "Date": datetime.date.today().isoformat()})
node = g
for i, n in enumerate(notes):
    node = node.variations[0]
    node.comment = n
open(f"games/{tag}.pgn", "w").write(str(g) + "\n")
outcome = b.outcome(claim_draw=True)
if getattr(llm, "violations", None):
    print(f"VIOLATIONS {tag}: {llm.violations}", flush=True)
print(f"RESULT {tag}: {result} ({outcome.termination.name if outcome else 'ply cap'})", flush=True)
q.close()
