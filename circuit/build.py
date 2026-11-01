"""The chess search machine as a circuit in the repo's q35 IR (fp32 exact).

Built in stages; each stage's dims are checked against the spec (machine/ref2.py) token by token
by test_circuit.py.

Stage 1: token features, the ply pointer (anchor + decay counters), the diff-stack board table
and the decoded board (absolute coordinates, one-hot per square).
"""
import fp32  # noqa: F401  (switches the IR's exactness checks to fp32)

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import A_BIT, A_REF, REF_COLS, Target
from chipformer.q35.prims import Circ, chan, sel, thead

import vocab as V
from ref import MAX_PLY

T4B = Target(key_heads=16, value_heads=32, intermediate=9216)
NCNT = 60  # decay counters are decoded up to this many events since the anchor
BIG = 1 << 12  # literal weight in fp32 threshold units (> any integer range they test)
DATA_COLS = list(range(2, 128))  # value columns a head row can hold data in


class ChessCirc(Circ):
    def __init__(self, target: Target | None = None) -> None:
        super().__init__(target or T4B)
        self.tables = {}  # name -> head node, for initial-state snapshots
        self.multi = {}  # name -> slots of tables read as a sum over rows (every row needs its ballast)
        self._features()

    # ------------------------------------------------------------------ token features

    def feat(self, name, fn):
        return self.token_feature(name, fn, full=True)

    def _features(self) -> None:
        def dec(t):
            return V.decode(t)

        kinds = ["START", "MOVE", "MOVE_IMP", "RET", "RET_ILLEGAL", "PLY", "PLY_IMP", "DONE"]
        self.is_ = {k: self.feat(f"is{k}", lambda t, k=k: int(dec(t)[0] == k)) for k in kinds}
        self.isMOVE = self.is_["MOVE"] + self.is_["MOVE_IMP"]
        self.isRET = self.is_["RET"] + self.is_["RET_ILLEGAL"]
        self.isPLY = self.is_["PLY"] + self.is_["PLY_IMP"]
        self.isANCHOR = self.is_["START"] + self.isPLY

        def ply_bit(i):
            def fn(t):
                d = dec(t)
                return (d[1] >> i & 1) if d[0] in ("PLY", "PLY_IMP") else 0
            return fn

        self.anchor_bits = [self.feat(f"anchor.b{i}", ply_bit(i)) for i in range(5)]

        def move_field(i):
            def fn(t):
                d = dec(t)
                return d[i] if d[0] in ("MOVE", "MOVE_IMP") else None
            return fn

        def onehot(name, getter, n):
            return [self.feat(f"{name}[{v}]", lambda t, v=v: int(getter(t) == v)) for v in range(n)]

        self.mv_f = onehot("mv.f", move_field(1), 64)
        self.mv_t = onehot("mv.t", move_field(2), 64)
        self.mv_pk = onehot("mv.pk", move_field(3), 7)
        self.mv_xk = onehot("mv.xk", move_field(4), 7)
        self.mv_white = self.feat("mv.white", lambda t: int(move_field(5)(t) == 1))

        # the move's DIFF row (absolute): per square, the code bits removed / added, rights lost
        def diff_of(t):
            d = dec(t)
            if d[0] not in ("MOVE", "MOVE_IMP"):
                return None
            _, f, to, pk, xk, white = d
            return _diff(f, to, pk, xk, bool(white))

        self._diff_cache = {}

        def diff_bit(kind, s, i):
            def fn(t):
                if t not in self._diff_cache:
                    self._diff_cache[t] = diff_of(t)
                d = self._diff_cache[t]
                if d is None:
                    return 0
                if kind == "lost":
                    return d["lost"] >> i & 1
                return d[kind].get(s, 0) >> i & 1
            return fn

        self.d_add = [[self.feat(f"d.add[{s}].b{i}", diff_bit("add", s, i)) for i in range(4)]
                      for s in range(64)]
        self.d_rem = [[self.feat(f"d.rem[{s}].b{i}", diff_bit("rem", s, i)) for i in range(4)]
                      for s in range(64)]
        self.d_lost = [self.feat(f"d.lost.b{i}", diff_bit("lost", 0, i)) for i in range(4)]

    # ------------------------------------------------------------------ fp32 gadgets

    def ge_terms(self, z, theta, lits=(), coef=1):
        """coef * [z >= theta] AND lits, for an integer z (fp32: any range below BIG)."""
        lits = list(lits)
        zz = z + BIG * (sum(lits, Expr()) - len(lits)) if lits else z
        return [(coef, self.one, zz - theta + 1), (-coef, self.one, zz - theta)]

    def eq_terms(self, z, v, lits=(), coef=1):
        return self.ge_terms(z, v, lits, coef) + self.ge_terms(z, v + 1, lits, -coef)

    # ------------------------------------------------------------------ counters (from rv/core.py)

    def counter(self, name, reset, step):
        """A 2^-k counter head: k = the `step` tokens since the latest `reset` token (0 there)."""
        one = self.one
        ((x,),) = thead(self, name, {0: reset, 1: one - reset}, {0: one - reset, 1: one - reset},
                        [{"data": {2: reset}, "ref": one - reset, "a": 110 * step + 238 * reset}],
                        tiny=2.0**-90)
        self.tables[name] = self.nodes[-1]
        self.nodes[-1].levels = "decay"
        return x

    def thermo(self, x, reset, n, name):
        """[k > m] for m < n, from x = 2^-k (0 at reset tokens)."""
        one = self.one
        return [self.node([(1, one - reset, 7 * one - 8 * 2**m * x),
                           (-1, one - reset, 6 * one - 8 * 2**m * x)], f"{name}>{m}")
                for m in range(n)]

    # ------------------------------------------------------------------ heads

    def multi_read(self, name, slots, keys, data_cols, ref, beta=None, rows_read=None):
        """A table whose reads sum every slot in ``slots`` (each row keeps its ballast, so the
        sum is normalised by the slot count, a power of 2). keys: slot -> 0/1 form; data_cols:
        column -> 0/1 form (one value head, <= 126 columns). Returns one dim per data column
        holding the column's sum over the rows."""
        n = len(slots) if rows_read is None else rows_read
        assert n & (n - 1) == 0, "the slot count must be a power of 2"
        v = {col: chan(e, A_BIT) for col, e in data_cols.items()}
        for col in REF_COLS:
            v[col] = chan(ref, A_REF)
        cols = sorted(data_cols)
        out = self.head(name, {s: sel(self, self.one) for s in slots},
                        {s: sel(self, e) for s, e in keys.items()}, v, 0,
                        {col: 128 * self.one for col in cols}, [(col, 2.0**7 * n) for col in cols],
                        beta=beta)
        self.tables[name] = self.nodes[-1]
        self.multi[name] = list(slots)
        return out


def _diff(f, t, pk, xk, white):
    """{square: code} removed / added by a move and the rights it loses (codes 1-6 White,
    7-12 Black). Mirrors machine/ref2.move_diff."""
    import ref2
    d = ref2.move_diff(f, t, pk, xk, white)
    return {"rem": {s: c for s, c in d["rem"]}, "add": {s: c for s, c in d["add"]}, "lost": d["lost"]}


def build_stage1(c: ChessCirc) -> dict:
    one, is_ = c.one, c.is_
    out = {}

    # ---------------------------------------------------------------- ply pointer
    (anchor_bits,) = thead(c, "anchor", {0: one}, {0: one},
                           [{"data": {2 + i: c.anchor_bits[i] for i in range(5)}, "ref": one,
                             "beta": c.isANCHOR}])
    c.tables["anchor"] = c.nodes[-1]
    xm = c.counter("cnt.move", c.isANCHOR, c.isMOVE)
    xr = c.counter("cnt.ret", c.isANCHOR, c.isRET)
    tm = c.thermo(xm, c.isANCHOR, NCNT, "cm")
    tr = c.thermo(xr, c.isANCHOR, NCNT, "cr")
    P = sum((anchor_bits[i] * (1 << i) for i in range(5)), Expr()) + sum(tm, Expr()) - sum(tr, Expr())
    p_oh = [c.node(c.eq_terms(P, r), f"P=={r}") for r in range(MAX_PLY)]
    # DIFF row keys: the child's row on MOVE (P), the returning child's on RET (P + 1)
    kd = [c.node(c.eq_terms(P, r, lits=[c.isMOVE]) + c.eq_terms(P, r - 1, lits=[c.isRET]), f"kd[{r}]")
          for r in range(MAX_PLY)]
    out.update(P=P, p_oh=p_oh)

    # ---------------------------------------------------------------- the diff-stack board
    # columns: per square 4 add bits + 4 rem bits (512) and 4 lost bits, split over value heads
    cols = []
    for s in range(64):
        for i in range(4):
            cols.append(("add", s, i, c.d_add[s][i]))
            cols.append(("rem", s, i, c.d_rem[s][i]))
    for i in range(4):
        cols.append(("lost", 0, i, c.d_lost[i]))
    sums = {}
    keys = {r: kd[r] for r in range(MAX_PLY)}
    for h in range(0, len(cols), len(DATA_COLS)):
        chunk = cols[h:h + len(DATA_COLS)]
        data = {DATA_COLS[j]: e for j, (_, _, _, e) in enumerate(chunk)}
        res = c.multi_read(f"diff{h // len(DATA_COLS)}", list(range(MAX_PLY)), keys, data, one)
        for (kind, s, i, _), d in zip(chunk, res):
            sums[kind, s, i] = d
    code = [sum(((sums["add", s, i] - sums["rem", s, i]) * (1 << i) for i in range(4)), Expr())
            for s in range(64)]
    board = [[c.node(c.eq_terms(code[s], k), f"abs[{s}]=={k}") for k in range(1, 13)] for s in range(64)]
    lost = [sums["lost", 0, i] for i in range(4)]
    out.update(code=code, board=board, lost=lost)
    return out


# ====================================================================== stage 2

KNIGHT_D = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
KING_D = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
DIAG_D = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ORTH_D = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def sq_at(r, f):
    return r * 8 + f if 0 <= r < 8 and 0 <= f < 8 else None


def attack_pairs():
    """(attacker square a, target s, class, squares between) for every geometric attack.
    class: 'N', 'K', 'D' (bishop/queen), 'O' (rook/queen)."""
    out = []
    for a in range(64):
        r, f = divmod(a, 8)
        for dr, df in KNIGHT_D:
            s = sq_at(r + dr, f + df)
            if s is not None:
                out.append((a, s, "N", []))
        for dr, df in KING_D:
            s = sq_at(r + dr, f + df)
            if s is not None:
                out.append((a, s, "K", []))
        for cls, dirs in (("D", DIAG_D), ("O", ORTH_D)):
            for dr, df in dirs:
                between = []
                for k in range(1, 8):
                    s = sq_at(r + dr * k, f + df * k)
                    if s is None:
                        break
                    out.append((a, s, cls, list(between)))
                    between.append(s)
    return out


PAIRS = attack_pairs()


def build_stage2(c: ChessCirc, s1: dict) -> dict:
    one = c.one
    is_ = c.is_
    p_oh, board = s1["p_oh"], s1["board"]
    out = {}

    # ---------------------------------------------------------------- root info and en passant
    def start_field(i):
        def fn(t):
            d = V.decode(t)
            return d[i] if d[0] == "START" else None
        return fn

    depth_bits = [c.feat(f"st.depth.b{i}", lambda t, i=i: (start_field(1)(t) or 0) >> i & 1) for i in range(4)]
    st_white = c.feat("st.white", lambda t: int(start_field(2)(t) == 1))
    castle_bits = [c.feat(f"st.castle.b{i}", lambda t, i=i: (start_field(3)(t) or 0) >> i & 1) for i in range(4)]
    (root,) = thead(c, "root", {0: one}, {0: one},
                    [{"data": {2 + i: e for i, e in enumerate(depth_bits + [st_white] + castle_bits)},
                      "ref": one, "beta": is_["START"]}])
    c.tables["root"] = c.nodes[-1]
    root_depth = sum((root[i] * (1 << i) for i in range(4)), Expr())
    root_white = root[4]
    root_castle = root[5:9]

    def ep_file_of(t):
        d = V.decode(t)
        if d[0] == "START":
            return d[4]
        if d[0] in ("MOVE", "MOVE_IMP"):
            _, f, to, pk, xk, w = d
            return f % 8 if pk == 1 and abs(to - f) == 16 else 8
        return 8

    ep_feat = [c.feat(f"ep.file=={x}", lambda t, x=x: int(ep_file_of(t) == x)) for x in range(8)]
    ep_key = {r: c.node(c.t_and([p_oh[r], c.isMOVE]) + (c.t_and([is_["START"]]) if r == 0 else []),
                        f"kep[{r}]") for r in range(MAX_PLY)}
    (ep_read,) = thead(c, "ep", ep_key, {r: p_oh[r] for r in range(MAX_PLY)},
                       [{"data": {2 + x: ep_feat[x] for x in range(8)}, "ref": one}])
    c.tables["ep"] = c.nodes[-1]

    # ---------------------------------------------------------------- side to move, relative board
    odd = sum((p_oh[r] for r in range(1, MAX_PLY, 2)), Expr())
    stm_white = c.node(c.t_and([root_white, one - odd]) + c.t_and([one - root_white, odd]), "stm.white")
    stm_black = one - stm_white

    def swap(k):  # absolute code -> the other colour's code
        return k + 6 if k <= 6 else k - 6

    # rel[s][k-1]: us = 1..6, them = 7..12 (relative to the side to move; Black is mirrored)
    rel = [[c.node(c.t_and([board[s][k - 1], stm_white]) + c.t_and([board[s ^ 56][swap(k) - 1], stm_black]),
                   f"rel[{s}]=={k}") for k in range(1, 13)] for s in range(64)]
    empty = [one - sum(rel[s], Expr()) for s in range(64)]
    us_any = [sum(rel[s][:6], Expr()) for s in range(64)]
    them_any = [sum(rel[s][6:], Expr()) for s in range(64)]

    # castling rights (absolute: 1 WK, 2 WQ, 4 BK, 8 BQ) -> relative (1 our O-O, 2 our O-O-O, ...)
    alive = [c.node(c.ge_terms(-s1["lost"][i], 0, lits=[root_castle[i]]), f"right[{i}]") for i in range(4)]
    rel_right = [c.node(c.t_and([alive[i], stm_white]) + c.t_and([alive[i ^ 2], stm_black]), f"relright[{i}]")
                 for i in range(4)]

    # ---------------------------------------------------------------- attack maps (relative)
    P_, N_, B_, R_, Q_, K_ = range(6)

    def piece(s, side, kind):  # side 0 us, 1 them; kind 0..5
        return rel[s][kind + 6 * side]

    def cls_lit(a, side, cls):
        if cls == "N":
            return piece(a, side, N_)
        if cls == "K":
            return piece(a, side, K_)
        if cls == "D":
            return piece(a, side, B_) + piece(a, side, Q_)
        return piece(a, side, R_) + piece(a, side, Q_)

    atk = [[[] for _ in range(64)] for _ in range(2)]  # side -> target -> unit terms
    for a, s, cls, between in PAIRS:
        for side in (0, 1):
            atk[side][s] += c.t_and([cls_lit(a, side, cls)] + [empty[b] for b in between])
    for a in range(64):
        r, f = divmod(a, 8)
        for side, dr in ((0, 1), (1, -1)):
            for df in (-1, 1):
                s = sq_at(r + dr, f + df)
                if s is not None:
                    atk[side][s] += c.t_and([piece(a, side, P_)])
    atk_us = [c.node(atk[0][s], f"atk.us[{s}]") for s in range(64)]
    atk_them = [c.node(atk[1][s], f"atk.them[{s}]") for s in range(64)]

    illegal = c.node([t for s in range(64) for t in c.ge_terms(atk_us[s], 1, lits=[piece(s, 1, K_)])], "illegal")
    in_check = c.node([t for s in range(64) for t in c.ge_terms(atk_them[s], 1, lits=[piece(s, 0, K_)])], "in_check")
    out.update(root_depth=root_depth, root_white=root_white, ep_read=ep_read, stm_white=stm_white,
               rel=rel, empty=empty, us_any=us_any, them_any=them_any, rel_right=rel_right,
               atk_us=atk_us, atk_them=atk_them, illegal=illegal, in_check=in_check)
    return out


# ====================================================================== stage 3

OFF = 32768  # scores are stored as x = score + OFF in 16 bits; x = 0 (score -32768) is -INF
DIV_M = 16384  # floor division offset: e24 + 24 * DIV_M >= 0


def int_bits16(c: ChessCirc, x, name: str) -> list:
    """The 16 bits of an integer x in [0, 65536), in two threshold stages (hi byte, then bits)."""
    one = c.one
    hi = c.node([t for j in range(1, 256) for t in c.ge_terms(x, 256 * j)], f"{name}.hi")
    lo = x - 256 * hi

    def byte_bits(z, tag):
        out = []
        for i in range(8):
            terms = []
            for theta in range(1, 256):
                d = (theta >> i & 1) - ((theta - 1) >> i & 1)
                if d:
                    terms += c.ge_terms(z, theta, coef=d)
            out.append(c.node(terms, f"{name}.{tag}{i}"))
        return out

    return byte_bits(lo, "b") + byte_bits(hi, "b8+")


def score_table(c: ChessCirc, name: str, rows: list, keys: dict, row_bits: dict, lag=False) -> dict:
    """Per-row 16-bit values in per-row column blocks, read all at once: returns {row: value
    (score + OFF as a linear form)}. row_bits: row -> 16 forms (0/1), the bits that row's write
    stores (0 on tokens that do not write it)."""
    cols = [(r, i) for r in rows for i in range(16)]
    vals = {r: Expr() for r in rows}
    per = len(DATA_COLS)
    for h in range(0, len(cols), per):
        chunk = cols[h:h + per]
        data = {DATA_COLS[j]: row_bits[r][i] for j, (r, i) in enumerate(chunk)}
        res = c.multi_read(f"{name}{h // per}", rows, keys, data, c.one)
        for (r, i), d in zip(chunk, res):
            vals[r] = vals[r] + d * (1 << i)
    return vals


def build_stage3(c: ChessCirc, s1: dict, s2: dict) -> dict:
    import pesto
    one, is_ = c.one, c.is_
    P, p_oh = s1["P"], s1["p_oh"]
    rel = s2["rel"]
    fresh = is_["START"] + c.isMOVE
    out = {}

    # ---------------------------------------------------------------- evaluation (PeSTO, tapered)
    mg = Expr()
    eg = Expr()
    ph = Expr()
    for s in range(64):
        for k in range(6):
            mg = mg + rel[s][k] * pesto.MG[k][s] - rel[s][k + 6] * pesto.MG[k][s ^ 56]
            eg = eg + rel[s][k] * pesto.EG[k][s] - rel[s][k + 6] * pesto.EG[k][s ^ 56]
            w = (0, 1, 1, 2, 4, 0)[k]
            if w:
                ph = ph + (rel[s][k] + rel[s][k + 6]) * w
    ph_oh = [c.node(c.eq_terms(ph, k), f"ph=={k}") for k in range(24)] + \
            [c.node(c.ge_terms(ph, 24), "ph>=24")]
    e24 = c.node([(1, mg * k + eg * (24 - k), ph_oh[k]) for k in range(25)], "e24")
    x = e24 + 24 * DIV_M  # >= 0
    q1 = c.node([t for j in range(1, 512) for t in c.ge_terms(x, 1536 * j)], "div.q1")
    r1 = x - 1536 * q1
    q2 = c.node([t for j in range(1, 64) for t in c.ge_terms(r1, 24 * j)], "div.q2")
    evalv = q1 * 64 + q2 - DIV_M
    out.update(mg=mg, eg=eg, ph=ph, eval=evalv)

    # ---------------------------------------------------------------- depth: check extensions
    plycap = p_oh[MAX_PLY - 1]
    ext_flag = c.node(c.t_and([s2["in_check"], one - s2["illegal"], one - plycap, fresh]), "ext.flag")
    k_fresh = {r: c.node(c.t_and([fresh, p_oh[r]]), f"kfresh[{r}]") for r in range(MAX_PLY)}
    k_ext = {r: c.node(c.t_and([fresh, p_oh[r]]) + (c.t_and([c.isRET, p_oh[r - 1]]) if r else []),
                       f"kext[{r}]") for r in range(MAX_PLY)}
    (ext_sum,) = c.multi_read("ext", list(range(MAX_PLY)), k_ext, {2: ext_flag}, one)
    depth = s2["root_depth"] - P + ext_sum
    out.update(depth=depth, ext_flag=ext_flag)

    # ---------------------------------------------------------------- stand pat (fresh quiescence nodes)
    qnode = c.node(c.ge_terms(-depth, 0, lits=[one - s2["illegal"], one - plycap, fresh]), "qnode")
    xs = c.node([(1, evalv + OFF, qnode)], "stand.x")  # 0 (= -INF) unless a quiescence node
    xs_bits = int_bits16(c, xs, "stand")
    st_bits = {r: [c.node(c.t_and([xs_bits[i], p_oh[r], fresh]), f"stand[{r}].b{i}") for i in range(16)]
               for r in range(MAX_PLY)}
    stand = score_table(c, "stand", list(range(MAX_PLY)), k_fresh, st_bits)
    out.update(qnode=qnode, stand_x=xs)

    # ---------------------------------------------------------------- best (by parity), lag head for v
    def ret_vbit(i):
        def fn(t):
            d = V.decode(t)
            return ((-d[1] + OFF) >> i & 1) if d[0] == "RET" else 0
        return fn

    vbits = [c.feat(f"ret.v.b{i}", ret_vbit(i)) for i in range(16)]
    from chipformer.q35.ir import Chan
    (prev_v,) = thead(c, "prev.v", {0: one}, {0: one},
                      [{"data": {2 + i: Chan(vbits[i], (0.0, 1.0, 0.0, 0.0)) for i in range(16)}, "ref": one}])
    c.tables["prev.v"] = c.nodes[-1]
    imp_move, imp_ply = is_["MOVE_IMP"], is_["PLY_IMP"]
    best = {}
    for par in (0, 1):
        rows = list(range(par, MAX_PLY, 2))
        keys, bits = {}, {}
        for r in rows:
            nxt = p_oh[r + 1] if r + 1 < MAX_PLY else Expr()
            imp_row = c.t_and([imp_move, nxt]) + c.t_and([imp_ply, p_oh[r]])
            keys[r] = c.node(c.t_and([fresh, p_oh[r]]) + imp_row, f"kbest[{r}]")
            bits[r] = [c.node((c.t_and([prev_v[i], imp_move, nxt]) if r + 1 < MAX_PLY else [])
                              + c.t_and([prev_v[i], imp_ply, p_oh[r]]), f"best[{r}].b{i}") for i in range(16)]
        best.update(score_table(c, f"best{par}.", rows, keys, bits))

    # ---------------------------------------------------------------- the window from the ancestors
    # (offset values: 0 = -INF). alpha: same parity as P, rows <= P; beta: other parity, rows < P.
    def incl(r, same):
        return sum((p_oh[p] for p in range(r, MAX_PLY) if (p - r) % 2 == (0 if same else 1)), Expr())

    def masked(r, same, tag):
        b_, s_ = best[r], stand[r]
        return c.node([(1, incl(r, same), b_), (1, incl(r, same), s_ - b_)], f"{tag}.m[{r}]")

    def max_tree(vals, tag):
        lvl = 0
        while len(vals) > 1:
            nxt = []
            for i in range(0, len(vals), 2):
                if i + 1 < len(vals):
                    a, b = vals[i], vals[i + 1]
                    nxt.append(c.node([(1, one, a), (1, one, b - a)], f"{tag}.max{lvl}.{i // 2}"))
                else:
                    nxt.append(vals[i])
            vals, lvl = nxt, lvl + 1
        return vals[0]

    am = [masked(r, True, "alpha") for r in range(MAX_PLY)]
    bm = [masked(r, False, "beta") for r in range(MAX_PLY)]
    alpha_x = max_tree(am[0::2], "alpha.e") + max_tree(am[1::2], "alpha.o")
    beta_x = max_tree(bm[0::2], "beta.e") + max_tree(bm[1::2], "beta.o")
    alpha = alpha_x - OFF
    beta = OFF - beta_x  # -(max - OFF)
    out.update(best=best, stand=stand, alpha=alpha, beta=beta)
    return out
