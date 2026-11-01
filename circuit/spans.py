"""fp32 sum-span check: the hardware sums every projection and every MLP output in some order,
so each such sum is exact in fp32 only if the sum of the absolute values of its terms stays
below 2^24 (all terms here are integers, or multiples of a common power of 2). The IR evaluator
sums in float64 and cannot see a violation; this measures the spans on an evaluated stream.

    spans(c, mat) -> [(span, node name, what)] over the limit, worst first
"""
import numpy as np

from chipformer.dsl.graph import ONE
from chipformer.q35.ir import FFN, Head

LIMIT = 2.0**24


def _span(e, mat):
    s = np.zeros(mat.shape[0])
    for d, w in e.terms.items():
        s += np.abs(float(w) * (1.0 if d == ONE else mat[:, d]))
    return s


def spans(c, mat, limit=LIMIT):
    bad = []
    out_sum = {}  # (sublayer, out dim) -> sum of |unit outputs|
    for n in c.nodes:
        if isinstance(n, FFN):
            acc = np.zeros(mat.shape[0])
            for coef, a, b in n.terms:
                sb, sa = _span(b, mat), _span(a, mat)
                av = sum((float(w) * (1.0 if d == ONE else mat[:, d]) for d, w in a.terms.items()),
                         np.zeros(mat.shape[0]))
                bv = sum((float(w) * (1.0 if d == ONE else mat[:, d]) for d, w in b.terms.items()),
                         np.zeros(mat.shape[0]))
                # a gate input far below 0 may round: relu still gives exactly 0
                risky = (sb >= limit) & (bv > -sb * 2.0**-16)
                if risky.any():
                    bad.append((sb[risky].max(), n.name, "gate input"))
                risky = (sa >= limit) & (np.maximum(bv, 0) > 0)
                if risky.any():
                    bad.append((sa[risky].max(), n.name, "up input"))
                acc += np.abs(float(coef) * av * np.maximum(bv, 0))
            key = (n.level, n.out)
            out_sum[key] = out_sum.get(key, 0) + acc
        elif isinstance(n, Head):
            for chans in (n.q, n.k):
                for ch in chans.values():
                    s = _span(ch.expr, mat).max()
                    if s >= limit:
                        bad.append((s, n.name, "q/k"))
            for vh in n.vheads():
                for ch in vh.v.values():
                    s = _span(ch.expr, mat).max()
                    # 0/1 forms times A_BIT / A_REF: at most two terms of one power of 2 are exact
                    if s > limit and len(ch.expr.terms) > 2:
                        bad.append((s, n.name, "v"))
    names = {}
    for n in c.nodes:
        if isinstance(n, FFN):
            names.setdefault((n.level, n.out), n.name)
    for key, acc in out_sum.items():
        if acc.max() >= limit:
            bad.append((acc.max(), names[key], "unit outputs"))
    return sorted(bad, reverse=True)
