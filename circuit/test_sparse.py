"""sparse_exec.SparseExec == ir.evaluate on every dim at every position of a bridge stream."""
import sys
import time

import numpy as np

import build as B
import test_bridge as TB
from chipformer.q35 import ir
from sparse_exec import SparseExec
from text import Text

depth = int(sys.argv[1]) if len(sys.argv) > 1 else 2
T = Text()
c, _ = B.build_all(T, depth)
toks = []
for i, fen in enumerate(["r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
                         "r3k2r/ppp2ppp/2nq1n2/3pp3/1b1PP1b1/2N2N2/PPPQBPPP/R3K2R b Qk d3 0 8"]):
    t, o, m, mv = TB.exchange(T, fen, depth, "<|im_start|>assistant\n" if i == 0 else " ok.\n")
    toks += t
want = ir.evaluate(c, toks, check=False)
t0 = time.time()
ex = SparseExec(c)
print(f"built in {time.time() - t0:.0f}s", flush=True)
ex.reset()
ex.step(toks[0])  # numba compile
ex.reset()
for t in set(toks):
    ex.embed(t)
tiny = 2.0**-90
bad = 0
t0 = time.time()
got_all = np.zeros((len(toks), ex.D), np.float32)
for i, t in enumerate(toks):
    got_all[i] = ex.step(t)
dt = (time.time() - t0) / len(toks)
for i in range(len(toks)):
    g = got_all[i].astype(np.float64)
    d = np.nonzero((g != want[i]) & ~((np.abs(g) <= tiny) & (np.abs(want[i]) <= tiny)))[0]
    if len(d):
        bad += 1
        if bad <= 3:
            print("  differs at", i, [(c.dims[k].name, g[k], want[i, k]) for k in d[:6]])
print(f"SPARSE: {len(toks)} positions, {bad} differ, {dt * 1000:.3f} ms/token")
