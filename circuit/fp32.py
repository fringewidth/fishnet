"""Use the repo's circuit IR (experimental/riscv/chipformer) with fp32 as the exact dtype.

The IR's evaluator checks every materialized value against a rounding function, bf16 by default.
The chess layers run in fp32, so the rounding becomes fp32's: every check ("this value survives
the hardware's rounding unchanged", "silu is an exact relu here") then holds for fp32. Integers
below 2^24 are exact, which is what the chess arithmetic uses.
"""
import os
import sys

import numpy as np

_VENDORED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "qwenputer",
                         "experimental", "riscv")
QWENPUTER = os.environ.get("QWENPUTER") or (_VENDORED if os.path.isdir(_VENDORED) else
                                             os.path.expanduser("~/qwenputer/experimental/riscv"))
if QWENPUTER not in sys.path:
    sys.path.insert(0, QWENPUTER)

from chipformer.q35 import ir  # noqa: E402


def _fp32(x):
    return np.asarray(x, dtype=np.float32).astype(np.float64)


ir.bf16 = _fp32  # is_bf16 / dead / eval_* all route through this name


# The emitter (q35/emit.py) checks every weight is a bf16 value and splits constants into bf16
# parts; with fp32 chip layers both become fp32's. (Only when torch is there: the IR tests run
# without it.)
try:
    import torch as _torch
    from chipformer.q35 import emit as _emit
except ImportError:
    _emit = None


def _check_fp32(name, t):
    b = t.to(_torch.float32).to(t.dtype)
    bad = ((b != t) | ~_torch.isfinite(b)).nonzero()
    if len(bad):
        i = tuple(bad[0].tolist())
        raise ValueError(f"{name}: weight {t[i].item()!r} at {i} is not an fp32 value")


def _fp32_parts(x, n):
    parts = []
    for _ in range(n):
        if x == 0:
            break
        p = float(np.float32(x))
        parts.append(p)
        x -= p
    if x != 0:
        raise ValueError(f"constant needs more than {n} fp32 parts")
    return parts


if _emit is not None:
    _emit.check_bf16 = _check_fp32
    _emit.bf16_parts = _fp32_parts
