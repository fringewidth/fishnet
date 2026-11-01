"""The reference search at a fixed depth against Stockfish at a limited Elo. Colours alternate
over a set of balanced openings. Prints W/D/L from the search's side and the implied Elo gap.

    uv run python harness/calibrate.py DEPTH SF_ELO GAMES [SF_SECONDS_PER_MOVE]

The reference plays exactly fishnet's moves (the weights are checked against it), so this
measures fishnet's strength without running the model.
"""
import math
import sys

import chess

from players import Reference, Stockfish

OPENINGS = ["", "e2e4 e7e5", "e2e4 c7c5", "d2d4 d7d5", "d2d4 g8f6 c2c4 e7e6", "c2c4 e7e5",
            "e2e4 e7e6", "e2e4 c7c6", "g1f3 d7d5", "d2d4 g8f6 c2c4 g7g6"]


def play(white, black, opening: str) -> str:
    b = chess.Board()
    for u in opening.split():
        b.push_uci(u)
    while not b.is_game_over(claim_draw=True) and b.ply() < 400:
        p = white if b.turn == chess.WHITE else black
        m = p.move(b)
        assert m in b.legal_moves, (p.name, m, b.fen())
        b.push(m)
    return b.result(claim_draw=True)


def main():
    depth, elo, games = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    sf_time = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    ref, sf = Reference(depth), Stockfish(elo, movetime=sf_time)
    w = d = lost = 0
    for g in range(games):
        opening = OPENINGS[(g // 2) % len(OPENINGS)]
        if g % 2 == 0:
            r = play(ref, sf, opening)
            s = {"1-0": 1, "0-1": 0}.get(r, 0.5)
        else:
            r = play(sf, ref, opening)
            s = {"1-0": 0, "0-1": 1}.get(r, 0.5)
        w += s == 1
        d += s == 0.5
        lost += s == 0
        print(f"game {g + 1}: {r} ({'W' if s == 1 else 'D' if s == 0.5 else 'L'})", flush=True)
    score = (w + d / 2) / games
    gap = -400 * math.log10(1 / min(max(score, 0.01), 0.99) - 1)
    print(f"depth {depth} vs Stockfish@{elo} ({sf_time}s/move): +{w} ={d} -{lost}  score {score:.2f}  "
          f"elo gap {gap:+.0f}")
    sf.close()


if __name__ == "__main__":
    main()
