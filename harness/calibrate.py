"""Native engine at a node budget vs Stockfish 16 at a limited Elo, alternating colours from a
set of balanced openings. Prints W/D/L from the engine's side and the implied Elo gap.
    calibrate.py NODES SF_ELO GAMES"""
import math
import sys

import chess

from players import Qwenputer, Stockfish

OPENINGS = ["", "e2e4 e7e5", "e2e4 c7c5", "d2d4 d7d5", "d2d4 g8f6 c2c4 e7e6", "c2c4 e7e5",
            "e2e4 e7e6", "e2e4 c7c6", "g1f3 d7d5", "d2d4 g8f6 c2c4 g7g6"]


def play(white, black, opening):
    b = chess.Board()
    for u in opening.split():
        b.push_uci(u)
    while not b.is_game_over(claim_draw=True) and b.ply() < 400:
        p = white if b.turn == chess.WHITE else black
        m = p.move(b)
        assert m in b.legal_moves, (p.name, m, b.fen())
        b.push(m)
    return b.result(claim_draw=True)


nodes, elo, games = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
q, sf = Qwenputer(nodes), Stockfish(elo)
w = d = l = 0
for g in range(games):
    op = OPENINGS[(g // 2) % len(OPENINGS)]
    if g % 2 == 0:
        r = play(q, sf, op)
        s = {"1-0": 1, "0-1": 0}.get(r, 0.5)
    else:
        r = play(sf, q, op)
        s = {"1-0": 0, "0-1": 1}.get(r, 0.5)
    w += s == 1
    d += s == 0.5
    l += s == 0
    print(f"game {g + 1}: {r} ({'W' if s == 1 else 'D' if s == 0.5 else 'L'})", flush=True)
score = (w + d / 2) / games
gap = -400 * math.log10(1 / min(max(score, 0.01), 0.99) - 1)
print(f"nodes={nodes} vs SF@{elo}: +{w} ={d} -{l}  score {score:.2f}  elo gap {gap:+.0f}")
q.close()
sf.close()
