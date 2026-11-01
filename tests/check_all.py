"""Bridge exchanges through the IR evaluator (exact next tokens), the fp32 span check and the
fast executor, on one stream.  uv run python tests/check_all.py DEPTH [N_EXEC]"""
import _paths  # noqa: F401  (circuit/ and machine/ on sys.path)
import sys
import time

import numpy as np

import build as B
import spans
import test_bridge as TB
from chipformer.q35 import ir
from text import Text

depth = int(sys.argv[1])
T = Text()
c, (s1, s2, s3, s4, s5, br, s6) = B.build_all(T, depth)
print("circuit", c.stats(), flush=True)
toks, own = [], []
for i, fen in enumerate(["r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
                         "r3k2r/ppp2ppp/2nq1n2/3pp3/1b1PP1b1/2N2N2/PPPQBPPP/R3K2R b Qk d3 0 8",
                         "8/P7/8/8/8/k7/8/7K w - - 0 1"]):
    t, o, m, mv = TB.exchange(T, fen, depth, "<|im_start|>assistant\n" if i == 0 else " ok.\n")
    toks += t
    own += o
t0 = time.time()
mat = ir.evaluate(c, toks, check=True)
code = np.stack([ir.ev(e, mat) for e in c.code], 1)
owner = ir.ev(s6["owner"], mat)
bad = 0
for i in range(len(toks) - 1):
    if own[i]:
        got = B.decode_code([int(v > 0) for v in code[i]], T) if owner[i] == 1 else None
        bad += got != toks[i + 1]
    else:
        bad += owner[i] != 0 or np.any(code[i] != 0)
print(f"IR: {len(toks)} positions, mismatches {bad} ({time.time() - t0:.0f}s)", flush=True)
sp = [x for x in spans.spans(c, mat) if not x[1].startswith("cnt.")]  # counters: multiples of 2^24
print("SPANS over 2^24:", len(sp), sp[:10], flush=True)
if len(sys.argv) > 2:
    from chipformer.rv import fastexec
    n = min(int(sys.argv[2]), len(toks))
    t0 = time.time()
    ex = fastexec.Executor(c, "cpu")
    ex.load_init()
    emb = ex.embed(toks[:n])
    tiny = 2.0**-90  # a halving counter far past its reset: denormal in one, 0 in the other
    nbad = 0
    for t in range(n):
        got = ex.step(emb[t:t + 1])[0].double().numpy()
        bad = np.nonzero((got != mat[t]) & ~((np.abs(got) <= tiny) & (np.abs(mat[t]) <= tiny)))[0]
        if len(bad):
            nbad += 1
            if nbad <= 3:
                print("  exec differs at", t, [(c.dims[d].name, got[d], mat[t, d]) for d in bad[:5]])
    print(f"EXEC: {n} positions, {nbad} differ ({time.time() - t0:.0f}s)", flush=True)
