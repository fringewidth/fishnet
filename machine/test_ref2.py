"""spec v2 (table layout) == spec v1 (verified search): same move, score, node count."""
import sys
import time

import ref
import ref2
from test_ref import positions

special = [
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
]
n, depth = int(sys.argv[1]), int(sys.argv[2])
bad, toks, nodes, t0 = 0, 0, 0, time.time()
for fen in special + positions(n, seed=3):
    pos, white = ref.from_fen(fen)
    m1 = ref.Machine(pos, depth)
    d1 = m1.run()
    m2 = ref2.Machine2(pos, white, depth)
    d2 = m2.run()
    bm = m2.bm
    got = ((ref2.to_abs(bm[1], True), ref2.to_abs(bm[2], True)) if bm else None, m2.nodes)
    want_move = (ref.real_square(d1[1], white), ref.real_square(d1[2], white)) if d1[1] >= 0 else None
    if got != (want_move, m1.nodes):
        bad += 1
        print("MISMATCH", fen, got, (want_move, m1.nodes))
    toks += len(m2.tokens)
    nodes += m2.nodes
print(f"depth {depth}: mismatches {bad} of {n + len(special)}, tokens/node {toks / max(nodes, 1):.3f}, "
      f"{time.time() - t0:.0f}s")
