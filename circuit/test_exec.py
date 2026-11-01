"""The repo's fast executor (rv/fastexec.py: the IR's arithmetic as batched torch ops) on the
chess circuit: every dim at every position equals ir.evaluate's, over a bridge exchange."""
import sys
import time

sys.argv += [] if len(sys.argv) > 1 else ["2"]
import build as B  # noqa: E402
import test_bridge as TB  # noqa: E402
from chipformer.rv import fastexec  # noqa: E402
from text import Text  # noqa: E402

T = Text()
c, _ = B.build_all(T, int(sys.argv[1]))
toks, own, m, mv = TB.exchange(T, "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
                               int(sys.argv[1]), "<|im_start|>assistant\n")
t0 = time.time()
n = fastexec.check(c, toks[:int(sys.argv[2]) if len(sys.argv) > 2 else None], device="cpu")
print(f"EXEC exact on {n} positions ({time.time() - t0:.0f}s)")
