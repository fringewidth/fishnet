"""Self-test of the chess layers on this machine, without Qwen.

Feeds the circuit a tool call holding a FEN, then lets it generate on its own: every token it
predicts is fed back in, as in a real game. The stream it writes (64 PUTs, START, the whole
search, DONE, the move printed as text) must equal the reference search machine's, token for
token.

    uv run python tests/selftest.py [DEPTH]
"""
import _paths  # noqa: F401  (circuit/ and machine/ on sys.path)
import sys
import time

import numpy as np

import build as B  # noqa: E402  (imports fp32 first)
import test_bridge as TB  # noqa: E402
import vocab as V  # noqa: E402
from sparse_exec import SparseExec  # noqa: E402
from text import Text  # noqa: E402

FENS = ["r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "r3k2r/ppp2ppp/2nq1n2/3pp3/1b1PP1b1/2N2N2/PPPQBPPP/R3K2R b Qk d3 0 8"]


def main() -> int:
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else B.DEPTH
    T = Text()
    t0 = time.time()
    c, _ = B.build_all(T, depth)
    ex = SparseExec(c)
    code_dims = np.array([next(iter(e.terms)) for e in c.code])
    print(f"built the chess layers in {time.time() - t0:.0f}s "
          f"({len(c.dims)} IR values, {sum(c.units.values())} MLP units, {sum(c.heads.values())} heads)")
    ex.reset()
    ex.step(V.TEXT0)  # numba compile
    failed = 0
    for i, fen in enumerate(FENS):
        want, own, _, move = TB.exchange(T, fen, depth, "<|im_start|>assistant\n")
        ex.reset()
        n_text = own.index(True) + 1  # Qwen's text up to and including </tool_call>
        got = list(want[:n_text])
        for t in got[:-1]:
            ex.step(t)
        t0 = time.time()
        while len(got) < len(want):
            x = ex.step(got[-1])
            code = x[code_dims]
            if not np.any(code != 0):
                break  # the circuit handed the stream back to Qwen
            got.append(B.decode_code([int(v > 0) for v in code], T))
        dt = time.time() - t0
        ok = got == want
        failed += not ok
        print(f"  position {i + 1}: {len(got) - n_text} tokens generated in {dt:.1f}s "
              f"({dt / max(1, len(got) - n_text) * 1000:.2f} ms/token), move {move}: "
              f"{'identical to the reference' if ok else 'DIFFERS from the reference'}")
        if not ok:
            j = next((k for k, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
            print(f"    first difference at token {j}: got {got[j:j + 3]}, want {want[j:j + 3]}")
    print("self-test passed" if not failed else "SELF-TEST FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
