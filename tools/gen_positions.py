"""Seeded random-play positions for equivalence checks: N games, sampling plies 0..80."""
import random
import sys

import chess

n = int(sys.argv[1])
rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
out = set()
while len(out) < n:
    b = chess.Board()
    for _ in range(rng.randrange(4, 80)):
        moves = list(b.legal_moves)
        if not moves:
            break
        caps = [m for m in moves if b.is_capture(m)]
        b.push(rng.choice(caps if caps and rng.random() < 0.3 else moves))
    if not b.is_game_over():
        out.add(b.fen())
print("\n".join(sorted(out)))
