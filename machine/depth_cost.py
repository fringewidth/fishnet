"""Nodes and tokens per search at fixed depths on sample positions."""
import statistics
import sys
import time

import ref
from test_ref import positions

fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"] + positions(5, seed=7)
for d in map(int, sys.argv[1:]):
    nodes, toks, t0 = [], [], time.time()
    for fen in fens:
        pos, _ = ref.from_fen(fen)
        m = ref.Machine(pos, d)
        m.run()
        nodes.append(m.nodes)
        toks.append(len(m.tokens))
    print(f"depth {d}: nodes median {statistics.median(nodes):.0f} max {max(nodes)}, "
          f"tokens/node {sum(toks) / max(sum(nodes), 1):.2f}, python {time.time() - t0:.0f}s for {len(fens)}")
