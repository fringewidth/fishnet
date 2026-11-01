"""Broader next-token check: mates, stalemate, random middlegames, depth 3."""
import _paths  # noqa: F401  (circuit/ and machine/ on sys.path)
import sys

sys.argv = sys.argv[:1]
import test_circuit as T  # noqa: E402
from test_ref import positions  # noqa: E402

cases = [("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", 3),            # back-rank mate in 1
         ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", 3),  # scholar's mate
         ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 3),                  # stalemate (black to move)
         ("k7/8/1K6/8/8/8/8/7R w - - 0 1", 3),                   # mate in 1 with rook
         ("8/P7/8/8/8/8/8/k6K w - - 0 1", 3)]                    # promotion
cases += [(f, 2) for f in positions(12, seed=11)]
cases += [(f, 3) for f in positions(3, seed=13)]
total = 0
for fen, d in cases:
    total += T.check_stage5(fen, d)
print("BROAD TOTAL mismatches", total, "over", len(cases), "searches")
