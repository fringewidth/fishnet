"""Use the repo's circuit IR (experimental/riscv/chipformer) with fp32 as the exact dtype.

The IR's evaluator checks every materialized value against a rounding function, bf16 by default.
The chess layers run in fp32, so the rounding becomes fp32's: every check ("this value survives
the hardware's rounding unchanged", "silu is an exact relu here") then holds for fp32. Integers
below 2^24 are exact, which is what the chess arithmetic uses.
"""
import os
import sys

import numpy as np

QWENPUTER = os.environ.get("QWENPUTER", os.path.expanduser("~/qwenputer/experimental/riscv"))
if QWENPUTER not in sys.path:
    sys.path.insert(0, QWENPUTER)

from chipformer.q35 import ir  # noqa: E402


def _fp32(x):
    return np.asarray(x, dtype=np.float32).astype(np.float64)


ir.bf16 = _fp32  # is_bf16 / dead / eval_* all route through this name
