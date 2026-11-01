"""A CPU executor for the circuit that skips zero inputs (numba).

The same arithmetic as the repo's rv/fastexec.py Executor (the IR's semantics in fp32: Gated
DeltaNet heads with one-hot writes and multi-hot reads normalized by the ballast, SwiGLU units
coef * a * relu(b)), for batch 1. Instead of multiplying every weight, an MLP sublayer scatters
only the inputs that are nonzero at this token: a token touches a few hundred of the circuit's
~11k dims, so most of the 1.6M MLP weights are never visited. Sums are taken in a different order
than the dense product, which is exact because every span is below 2^24 (spans.py).

    ex = SparseExec(c); ex.reset(); x = ex.step(token)   # x: every IR dim after the token
"""
import numba as nb
import numpy as np

from chipformer.dsl.graph import ONE
from chipformer.q35.ir import FFN, HD, REF_COLS, Head

F32 = np.float32
R0, R1 = REF_COLS[0], REF_COLS[-1]


class _Csr:
    """Rows of (src, weight) entries; src = lag * D + dim."""

    def __init__(self):
        self.ptr, self.src, self.w = [0], [], []

    def row(self, entries):
        for s, w in entries:
            self.src.append(s)
            self.w.append(w)
        self.ptr.append(len(self.src))

    def arrays(self):
        return (np.array(self.ptr, np.int64), np.array(self.src, np.int64), np.array(self.w, F32))


def _expr_entries(e, D, lag=0, tap=1.0):
    return [(lag * D + d, float(cf) * tap) for d, cf in e.terms.items()]


def _chan_entries(ch, D):
    out = []
    for lag, tw in enumerate(ch.taps):
        if tw:
            out += _expr_entries(ch.expr, D, lag, tw)
    return out


class SparseExec:
    def __init__(self, c):
        self.c = c
        D = self.D = len(c.dims)
        levels = sorted({n.level for n in c.nodes})
        kind, lv_a, lv_b = [], [], []  # per level: kind (0 ffn, 1 heads), ranges
        # FFN: referenced gate-input dims per level, their columns (unit, w); units: coef, out, a rows
        in_ptr, in_dim, col_ptr, col_u, col_w = [0], [], [0], [], []
        u_coef, u_out = [], []
        a_csr = _Csr()
        # heads: per level the key heads; per key head q / k rows (HD each); per value head
        # v rows (HD), z rows (HD), a row, beta row, bconst, out entries
        kh_ptr, kh_vptr, kh_lv = [0], [0], []
        qk = _Csr()
        vrows, zrows, arow, brow = _Csr(), _Csr(), _Csr(), _Csr()
        bconst, vh_kh = [], []
        out_ptr, out_col, out_dim, out_w = [0], [], [], []
        hist_slot = []  # per level: index of its input history (heads with lags), or -1
        n_hist = 0
        n_units = 0
        n_kh = n_vh = 0
        for lv in levels:
            nodes = [n for n in c.nodes if n.level == lv]
            if isinstance(nodes[0], FFN):
                kind.append(0)
                terms = [(cf, a, b, n.out) for n in nodes for cf, a, b in n.terms]
                cols: dict[int, list] = {}
                for j, (cf, a, b, out) in enumerate(terms):
                    u = n_units + j
                    for d, w in b.terms.items():
                        cols.setdefault(d, []).append((u, float(w)))
                    u_coef.append(float(cf))
                    u_out.append(out)
                    a_csr.row(_expr_entries(a, D))
                lv_a.append(n_units)
                n_units += len(terms)
                lv_b.append(n_units)
                for d in sorted(cols):
                    in_dim.append(d)
                    for u, w in cols[d]:
                        col_u.append(u)
                        col_w.append(w)
                    col_ptr.append(len(col_u))
                in_ptr.append(len(in_dim))
                hist_slot.append(-1)
                continue
            in_ptr.append(len(in_dim))
            kind.append(1)
            heads = [n for n in nodes if isinstance(n, Head)]
            lv_a.append(n_kh)
            uses_lag = False
            for n in heads:
                for chans in (n.q, n.k):
                    for s in range(HD):
                        ch = chans.get(s)
                        ent = _chan_entries(ch, D) if ch is not None else []
                        uses_lag |= any(e[0] >= D for e in ent)
                        qk.row(ent)
                for vd in n.vheads():
                    for col in range(HD):
                        ch = vd.v.get(col)
                        ent = _chan_entries(ch, D) if ch is not None else []
                        uses_lag |= any(e[0] >= D for e in ent)
                        vrows.row(ent)
                        e = vd.z.get(col)
                        zrows.row(_expr_entries(e, D) if e is not None else [])
                    arow.row(_expr_entries(vd.a, D))
                    brow.row(_expr_entries(vd.beta, D) if vd.beta is not None else [])
                    bconst.append(1.0 if vd.beta is None else 0.0)
                    vh_kh.append(n_kh)
                    for col, d, w in vd.out:
                        out_col.append(col)
                        out_dim.append(d)
                        out_w.append(w)
                    out_ptr.append(len(out_col))
                    n_vh += 1
                n_kh += 1
                kh_ptr.append(n_kh)
                kh_vptr.append(n_vh)
                kh_lv.append(lv)
            lv_b.append(n_kh)
            hist_slot.append(n_hist if uses_lag else -1)
            n_hist += 1 if uses_lag else 0
        self.n_levels = len(levels)
        self.kind = np.array(kind, np.int64)
        self.lv_a, self.lv_b = np.array(lv_a, np.int64), np.array(lv_b, np.int64)
        self.in_ptr, self.in_dim = np.array(in_ptr, np.int64), np.array(in_dim, np.int64)
        self.col_ptr, self.col_u, self.col_w = (np.array(col_ptr, np.int64), np.array(col_u, np.int64),
                                                np.array(col_w, F32))
        self.u_coef, self.u_out = np.array(u_coef, F32), np.array(u_out, np.int64)
        self.a_ptr, self.a_src, self.a_w = a_csr.arrays()
        self.kh_vptr = np.array(kh_vptr, np.int64)
        self.qk_ptr, self.qk_src, self.qk_w = qk.arrays()
        self.v_ptr, self.v_src, self.v_w = vrows.arrays()
        self.z_ptr, self.z_src, self.z_w = zrows.arrays()
        self.ar_ptr, self.ar_src, self.ar_w = arow.arrays()
        self.br_ptr, self.br_src, self.br_w = brow.arrays()
        self.bconst = np.array(bconst, F32)
        self.out_ptr, self.out_col = np.array(out_ptr, np.int64), np.array(out_col, np.int64)
        self.out_dim, self.out_w = np.array(out_dim, np.int64), np.array(out_w, F32)
        self.hist_slot = np.array(hist_slot, np.int64)
        self.S = np.zeros((n_vh, HD, HD), F32)
        self.hist = np.zeros((max(n_hist, 1), 4, D), F32)  # [slot, lag 0..3, dim]
        self.ub = np.zeros(max(n_units, 1), F32)
        self.x = np.zeros(D, F32)
        self.dx = np.zeros(D, F32)
        self.qkbuf = np.zeros((2, HD), F32)
        self.feat_ids = np.array(sorted(c.token_fns), np.int64)
        self.emb_cache: dict[int, tuple] = {}

    def reset(self) -> None:
        self.S[:] = 0
        self.hist[:] = 0

    def embed(self, t: int):
        e = self.emb_cache.get(t)
        if e is None:
            vals = np.array([float(self.c.feature_value(int(d), t)) for d in self.feat_ids], F32)
            nz = vals != 0
            e = (self.feat_ids[nz], vals[nz])
            self.emb_cache[t] = e
        return e

    def step(self, t: int) -> np.ndarray:
        idx, vals = self.embed(t)
        self.x[:] = 0
        self.x[ONE] = 1.0
        self.x[idx] = vals
        _step(self.x, self.dx, self.kind, self.lv_a, self.lv_b, self.in_ptr, self.in_dim, self.col_ptr,
              self.col_u, self.col_w, self.ub, self.u_coef, self.u_out, self.a_ptr, self.a_src, self.a_w,
              self.kh_vptr, self.qk_ptr, self.qk_src, self.qk_w, self.v_ptr, self.v_src, self.v_w,
              self.z_ptr, self.z_src, self.z_w, self.ar_ptr, self.ar_src, self.ar_w, self.br_ptr,
              self.br_src, self.br_w, self.bconst, self.out_ptr, self.out_col, self.out_dim, self.out_w,
              self.hist_slot, self.hist, self.S, self.qkbuf, self.D)
        return self.x


@nb.njit(cache=True, inline="always")
def _row(ptr, src, w, r, x, hist_l, D):
    s = np.float32(0.0)
    for i in range(ptr[r], ptr[r + 1]):
        j = src[i]
        if j < D:
            s += w[i] * x[j]
        else:
            lag = j // D
            s += w[i] * hist_l[lag - 1, j - lag * D]
    return s


@nb.njit(cache=True)
def _step(x, dx, kind, lv_a, lv_b, in_ptr, in_dim, col_ptr, col_u, col_w, ub, u_coef, u_out, a_ptr,
          a_src, a_w, kh_vptr, qk_ptr, qk_src, qk_w, v_ptr, v_src, v_w, z_ptr, z_src, z_w, ar_ptr,
          ar_src, ar_w, br_ptr, br_src, br_w, bconst, out_ptr, out_col, out_dim, out_w, hist_slot,
          hist, S, qkbuf, D):
    HDn = S.shape[1]
    ovec = np.zeros(HDn, np.float32)
    vvec = np.zeros(HDn, np.float32)
    for L in range(kind.shape[0]):
        if kind[L] == 0:
            u0, u1 = lv_a[L], lv_b[L]
            for u in range(u0, u1):
                ub[u] = 0.0
            for e in range(in_ptr[L], in_ptr[L + 1]):
                xv = x[in_dim[e]]
                if xv != 0.0:
                    for i in range(col_ptr[e], col_ptr[e + 1]):
                        ub[col_u[i]] += col_w[i] * xv
            # outputs from the pre-sublayer x, then added
            for u in range(u0, u1):
                b = ub[u]
                if b > 0.0:
                    a = np.float32(0.0)
                    for i in range(a_ptr[u], a_ptr[u + 1]):
                        a += a_w[i] * x[a_src[i]]
                    ub[u] = u_coef[u] * a * b
                else:
                    ub[u] = 0.0
            for u in range(u0, u1):
                if ub[u] != 0.0:
                    x[u_out[u]] += ub[u]
            continue
        # ---- a heads sublayer
        hs = hist_slot[L]
        hist_l = hist[max(hs, 0), 1:]  # lags 1..3 (only read when hs >= 0)
        for d in range(D):
            dx[d] = 0.0
        for kh in range(lv_a[L], lv_b[L]):
            for side in range(2):
                for s in range(HDn):
                    r = (2 * kh + side) * HDn + s
                    val = _row(qk_ptr, qk_src, qk_w, r, x, hist_l, D)
                    qkbuf[side, s] = val if val > 0.0 else 0.0
            kmax = np.float32(0.0)
            kidx = 0
            for s in range(HDn):
                if qkbuf[1, s] > kmax:
                    kmax = qkbuf[1, s]
                    kidx = s
            for vh in range(kh_vptr[kh], kh_vptr[kh + 1]):
                a = _row(ar_ptr, ar_src, ar_w, vh, x, hist_l, D)
                if a != 0.0:
                    fac = np.float32(0.5) if a == 110.0 else np.float32(0.0)
                    for r in range(HDn):
                        for col in range(HDn):
                            S[vh, r, col] *= fac
                beta = _row(br_ptr, br_src, br_w, vh, x, hist_l, D) + bconst[vh]
                if kmax > 0.0 and beta == 1.0:
                    for col in range(HDn):
                        val = _row(v_ptr, v_src, v_w, vh * HDn + col, x, hist_l, D)
                        S[vh, kidx, col] = val if val > 0.0 else 0.0
                for col in range(HDn):
                    ovec[col] = 0.0
                for r in range(HDn):
                    if qkbuf[0, r] > 0.0:
                        for col in range(HDn):
                            ovec[col] += S[vh, r, col]
                ref = np.float32(0.0)
                for col in range(R0, R1 + 1):
                    ref += ovec[col] * ovec[col]
                if ref > 0.0:
                    norm = np.sqrt(ref / np.float32(HDn))
                    for i in range(out_ptr[vh], out_ptr[vh + 1]):
                        col = out_col[i]
                        z = _row(z_ptr, z_src, z_w, vh * HDn + col, x, hist_l, D)
                        if z > 0.0 and ovec[col] != 0.0:
                            dx[out_dim[i]] += (ovec[col] / norm) * z * out_w[i]
        if hs >= 0:  # shift the input history: lag 3 <- 2 <- 1 <- this token
            for lag in range(3, 0, -1):
                for d in range(D):
                    hist[hs, lag, d] = hist[hs, lag - 1, d]
            for d in range(D):
                hist[hs, 1, d] = x[d]
        for d in range(D):
            if dx[d] != 0.0:
                x[d] += dx[d]
    return x
