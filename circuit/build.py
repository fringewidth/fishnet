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
BIG = 1 << 20  # literal weight in fp32 threshold units (> any integer range they test)
DATA_COLS = list(range(2, 128))  # value columns a head row can hold data in
BAL = 127  # the ballast row of a summed table


class ChessCirc(Circ):
    def __init__(self, target: Target | None = None, text=None) -> None:
        super().__init__(target or T4B)
        self.tables = {}  # name -> head node, for initial-state snapshots
        self.multi = {}  # name -> slots of tables read as a sum over rows (every row needs its ballast)
        self.text = text
        self._features()
        self._put_features()
        if text is not None:
            self._text_features()

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

    def _put_features(self) -> None:
        """PUT(s, code): the bridge's root-board tokens (and the square the next one names)."""
        def put(t):
            d = V.decode(t)
            return (d[1], d[2]) if d[0] == "PUT" else None

        self.isPUT = self.feat("isPUT", lambda t: int(put(t) is not None))
        self.put_sq = [self.feat(f"put.sq[{s}]", lambda t, s=s: int((put(t) or (-1, 0))[0] == s))
                       for s in range(64)]
        self.put_bit = [[self.feat(f"put.b{i}@{s}", lambda t, s=s, i=i: int((put(t) or (-1, 0))[0] == s
                                                                            and put(t)[1] >> i & 1))
                         for s in range(64)] for i in range(4)]

        def nxt(t):
            if self.text is not None and t == V.TEXT0 + self.text.TCC:
                return 0
            p = put(t)
            return p[0] + 1 if p is not None and p[0] < 63 else None

        tcc = V.TEXT0 + self.text.TCC if self.text is not None else -1
        self.isTCC = self.feat("isTCC", lambda t: int(t == tcc))
        self.put_next = [self.feat(f"put.next[{s}]", lambda t, s=s: int(nxt(t) == s)) for s in range(64)]
        self.put_last = self.feat("put.last", lambda t: int((put(t) or (-1, 0))[0] == 63))

    def _text_features(self) -> None:
        """Text tokens (TEXT0 + Qwen id): what the FEN reader needs from a token's bytes."""
        T, t0 = self.text, V.TEXT0
        info = T.fen_info

        def tinfo(t):
            return info.get(t - t0) if t >= t0 else None

        def flag(key):
            return lambda t: int(bool((tinfo(t) or {}).get(key, 0)))

        self.is_text = self.feat("is_text", lambda t: int(t >= t0))
        self.isTCO = self.feat("isTCO", lambda t: int(t == t0 + T.TCO))
        # a line starts after a newline (and at <tool_call>); counters reset there
        self.t_reset = self.feat("t.reset", lambda t: int(t == t0 + T.TCO or bool((tinfo(t) or {}).get("nl"))))

        def nr(fn):
            return lambda t: int(fn(t) and not (t == t0 + T.TCO or (tinfo(t) or {}).get("nl")))

        self.t_space = self.feat("t.space", nr(flag("space")))
        self.t_fenok = self.feat("t.fenok", flag("fenok"))
        self.t_notfen = self.feat("t.notfen", nr(lambda t: t >= t0 and not (tinfo(t) or {}).get("fenok")))
        self.t_boardok = self.feat("t.boardok", flag("boardok"))
        self.t_notboard = self.feat("t.notboard", nr(lambda t: t >= t0 and not (tinfo(t) or {}).get("boardok")))

        def codes(t):
            return (tinfo(t) or {}).get("codes") or []

        self.t_wb = [self.feat(f"t.w.b{b}", lambda t, b=b: len(codes(t)) >> b & 1) for b in range(4)]
        self.t_wgt = [self.feat(f"t.w>{k}", lambda t, k=k: int(len(codes(t)) > k)) for k in range(8)]
        self.t_code = [[self.feat(f"t.code{k}.b{i}", lambda t, k=k, i=i: (codes(t)[k] >> i & 1)
                                  if k < len(codes(t)) else 0) for i in range(4)] for k in range(8)]
        self.t_has = {ch: self.feat(f"t.has.{ch}", flag("has_" + ch)) for ch in "wbKQkq"}
        self.t_file = [self.feat(f"t.file{x}", lambda t, x=x: int((tinfo(t) or {}).get("file") == x))
                       for x in range(8)]
        # the output code (a token's own, in its embedding row; the exit overwrites it), mu, tau
        self._code_cache = {}

        def code_of(t):
            if t not in self._code_cache:
                self._code_cache[t] = token_code(t, T)
            return self._code_cache[t]

        self.code = [self.feat(f"code{i}", lambda t, i=i: 0 if (cb := code_of(t)) is None
                               else CODE_AMP * (2 * cb[i] - 1)) for i in range(NFAM + NPAY)]
        self.mu = self.feat("mu", lambda t: int(t < t0))  # Qwen is muted on chip tokens
        self.tau = self.feat("tau", lambda t: TAU_TEXT if t >= t0 else -1)

    # ------------------------------------------------------------------ fp32 gadgets

    def ge_terms(self, z, theta, lits=(), coef=1):
        """coef * [z >= theta] AND lits, for an integer z (fp32: any range below BIG)."""
        lits = list(lits)
        zz = z + BIG * (sum(lits, Expr()) - len(lits)) if lits else z
        return [(coef, self.one, zz - theta + 1), (-coef, self.one, zz - theta)]

    def digit(self, z, d, hi, name, nbits=0, fn=None, value=True):
        """q = floor(z / d) for an integer z in [0, hi] (sum of thresholds), and the bits of
        fn(q) from the same thresholds (no extra stage). Returns (q or None, bits)."""
        fn = fn or (lambda q: q)
        n = hi // d
        thr = [self.ge_terms(z, d * j) for j in range(1, n + 1)]
        q = self.node([t for ts in thr for t in ts], name) if value else None
        bits = []
        for i in range(nbits):
            terms = self.t_const(1) if fn(0) >> i & 1 else []
            for j in range(1, n + 1):
                dlt = (fn(j) >> i & 1) - (fn(j - 1) >> i & 1)
                if dlt:
                    terms += [(dlt * cf, a, b) for cf, a, b in thr[j - 1]]
            bits.append(self.node(terms, f"{name}.b{i}"))
        return q, bits

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

    def multi_read(self, name, slots, keys, data_cols, init, beta=None):
        """A table whose reads sum every row in ``slots``. Data rows carry no ballast; one extra
        row (BAL) holds it, written at the ``init`` token, so a read is the exact column sum of
        the data rows once that token has passed (and 0 before it). keys: slot -> 0/1 form;
        data_cols: column -> 0/1 form (one value head, <= 126 columns). Returns one dim per data
        column holding the column's sum over the rows."""
        assert BAL not in slots
        v = {col: chan(e, A_BIT) for col, e in data_cols.items()}
        for col in REF_COLS:
            v[col] = chan(init, A_REF)
        cols = sorted(data_cols)
        keys = dict(keys)
        keys[BAL] = init
        out = self.head(name, {s: sel(self, self.one) for s in [*slots, BAL]},
                        {s: sel(self, e) for s, e in keys.items()}, v, 0,
                        {col: 128 * self.one for col in cols}, [(col, 2.0**7) for col in cols],
                        beta=None if beta is None else beta + init)
        self.nodes[-1].qhot = len(slots) + 1
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
    out.update(P=P, p_oh=p_oh, tm=tm, tr=tr)

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
        res = c.multi_read(f"diff{h // len(DATA_COLS)}", list(range(MAX_PLY)), keys, data, c.put_sq[0])
        for (kind, s, i, _), d in zip(chunk, res):
            sums[kind, s, i] = d
    # the root board: one row per square, written by its PUT token (a bit plane per head)
    rb = [c.multi_read(f"rb{i}", list(range(64)), {s: c.put_sq[s] for s in range(64)},
                       {2 + s: c.put_bit[i][s] for s in range(64)}, c.isTCC, beta=c.isPUT) for i in range(4)]
    code = [sum(((sums["add", s, i] - sums["rem", s, i] + rb[i][s]) * (1 << i) for i in range(4)), Expr())
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
    """The 16 bits of an integer x in [0, 65536), in three digit stages (top nibble, next nibble,
    low byte; each stage's terms sum far below 2^24)."""
    n1, b_hi = c.digit(x, 4096, 65535, f"{name}.n1", nbits=4)
    r = x - 4096 * n1
    n2, b_mid = c.digit(r, 256, 4095, f"{name}.n2", nbits=4)
    _, b_lo = c.digit(r - 256 * n2, 1, 255, f"{name}.lo", nbits=8, value=False)
    return b_lo + b_mid + b_hi


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
        res = c.multi_read(f"{name}{h // per}", rows, keys, data, c.put_sq[0])
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
    # floor(x / 24) in three digit stages (98304 = 1536 * 64, 1536 = 24 * 64), so that every
    # stage's terms stay far below 2^24 in sum (fp32-exact in any summation order). q = x // 24
    # = 4096 q1 + 64 qm + q2, and the stand-pat value eval + OFF = q + 2^14 has bits q2 (0-5),
    # qm (6-11) and q1 + 4 (12-15), each from its own stage.
    x = e24 + 24 * DIV_M  # in [0, 2 * 24 * DIV_M)
    q1, xb_hi = c.digit(x, 98304, 2 * 24 * DIV_M - 1, "div.q1", nbits=4, fn=lambda q: q + 4)
    r1 = x - 98304 * q1
    qm, xb_mid = c.digit(r1, 1536, 98303, "div.qm", nbits=6)
    r2 = r1 - 1536 * qm
    q2, xb_lo = c.digit(r2, 24, 1535, "div.q2", nbits=6)
    evalv = 4096 * q1 + 64 * qm + q2 - DIV_M
    eval_bits = xb_lo + xb_mid + xb_hi  # bits of eval + OFF
    out.update(mg=mg, eg=eg, ph=ph, eval=evalv)

    # ---------------------------------------------------------------- depth: check extensions
    plycap = p_oh[MAX_PLY - 1]
    # (gated by `fresh` as a factor: the operands are not 0/1 on tokens before the tables are set up)
    ext_flag = c.node(c.t_and([s2["in_check"], one - s2["illegal"], one - plycap], a=fresh), "ext.flag")
    k_fresh = {r: c.node(c.t_and([fresh, p_oh[r]]), f"kfresh[{r}]") for r in range(MAX_PLY)}
    k_ext = {r: c.node(c.t_and([fresh, p_oh[r]]) + (c.t_and([c.isRET, p_oh[r - 1]]) if r else []),
                       f"kext[{r}]") for r in range(MAX_PLY)}
    (ext_sum,) = c.multi_read("ext", list(range(MAX_PLY)), k_ext, {2: ext_flag}, c.put_sq[0])
    depth = s2["root_depth"] - P + ext_sum
    out.update(depth=depth, ext_flag=ext_flag)

    # ---------------------------------------------------------------- stand pat (fresh quiescence nodes)
    qnode = c.node(c.ge_terms(-depth, 0, lits=[one - s2["illegal"], one - plycap, fresh]), "qnode")
    xs = c.node([(1, evalv + OFF, qnode)], "stand.x")  # 0 (= -INF) unless a quiescence node
    st_bits = {r: [c.node(c.t_and([eval_bits[i], qnode, p_oh[r]], a=fresh), f"stand[{r}].b{i}")
                   for i in range(16)]
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


# ====================================================================== stage 4: move selection

def kind_pairs():
    """(kind 1..6 [no pawns: 2..6], a, t, squares between) for every geometric non-pawn move."""
    out = []
    for a in range(64):
        r, f = divmod(a, 8)
        for k, dirs, reach in ((2, KNIGHT_D, 1), (6, KING_D, 1), (3, DIAG_D, 7), (4, ORTH_D, 7),
                               (5, DIAG_D + ORTH_D, 7)):
            for dr, df in dirs:
                between = []
                for d in range(1, reach + 1):
                    t = sq_at(r + dr * d, f + df * d)
                    if t is None:
                        break
                    out.append((k, a, t, list(between)))
                    between.append(t)
    return out


KPAIRS = kind_pairs()


def priority(c: ChessCirc, ind: list, name: str) -> list:
    """first[j] = ind[j] AND no ind[i] for i < j (ind: 0/1 forms)."""
    one = c.one
    out = []
    for j, e in enumerate(ind):
        if e is None:
            out.append(Expr())
            continue
        lits = [e] + [one - ind[i] for i in range(j) if ind[i] is not None]
        out.append(c.node(c.t_and(lits), f"{name}[{j}]"))
    return out


def build_stage4(c: ChessCirc, s1: dict, s2: dict, s3: dict) -> dict:
    one, is_ = c.one, c.is_
    p_oh = s1["p_oh"]
    rel, empty = s2["rel"], s2["empty"]
    stm_white = s2["stm_white"]
    stm_black = one - stm_white
    fresh = is_["START"] + c.isMOVE
    out = {}

    def us(s, k):  # our piece of kind k (1..6) on s
        return rel[s][k - 1]

    def them(s, k):
        return rel[s][k + 5]

    # ---------------------------------------------------------------- the cursor (last move tried)
    def mv_bits(i, lo, n):
        def fn(t):
            d = V.decode(t)
            return (d[i] >> lo & 1) if d[0] in ("MOVE", "MOVE_IMP") else 0
        return fn

    fields = [("f", 1, 6), ("t", 2, 6), ("pk", 3, 3), ("xk", 4, 3)]
    data, col = {}, 2
    names = []
    for nm, i, n in fields:
        for b in range(n):
            data[col] = c.feat(f"cur.{nm}.b{b}", mv_bits(i, b, n))
            names.append((nm, b))
            col += 1
    k_cur = {r: c.node(c.t_and([c.isMOVE, p_oh[r + 1]]), f"kcur[{r}]") for r in range(MAX_PLY - 1)}
    (cur,) = thead(c, "cursor", k_cur, {r: p_oh[r] for r in range(MAX_PLY - 1)},
                   [{"data": data, "ref": one}])
    c.tables["cursor"] = c.nodes[-1]
    cb = {nm: [None] * n for nm, _, n in fields}
    for (nm, b), d in zip(names, cur):
        cb[nm][b] = d
    has_cur = one - fresh  # a fresh node has no cursor (its row is stale)

    def rel_onehot(bits, tag):
        out_ = []
        for s in range(64):
            lw = [b if s >> i & 1 else one - b for i, b in enumerate(bits)]
            lb = [b if (s ^ 56) >> i & 1 else one - b for i, b in enumerate(bits)]
            out_.append(c.node(c.t_and(lw + [stm_white, has_cur]) + c.t_and(lb + [stm_black, has_cur]),
                               f"{tag}[{s}]"))
        return out_

    out["cur_bits"] = cb
    cf = rel_onehot(cb["f"], "cur.f")  # relative one-hots (all 0 without a cursor)
    ct = rel_onehot(cb["t"], "cur.t")
    pk_oh = [c.node(c.t_and([b if k >> i & 1 else one - b for i, b in enumerate(cb["pk"])] + [has_cur]),
                    f"cur.pk=={k}") for k in range(1, 7)]
    xk_oh = [c.node(c.t_and([b if k >> i & 1 else one - b for i, b in enumerate(cb["xk"])] + [has_cur]),
                    f"cur.xk=={k}") for k in range(0, 7)]
    f0 = sum((cf[s] * s for s in range(64)), Expr())
    t0 = sum((ct[s] * s for s in range(64)), Expr())
    cur_promo = c.node([t for s in range(56, 64) for t in c.t_and([ct[s], pk_oh[0], xk_oh[0]])], "cur.promo")
    cur_tac = has_cur - xk_oh[0] + cur_promo  # a capture, en passant or a queen promotion
    cur_quiet = has_cur - cur_tac
    # victim value of a tactical cursor: xk 1..5, en passant 1, promotion push 6
    cv_val = sum((xk_oh[k] * k for k in range(1, 6)), Expr()) + xk_oh[6] + cur_promo * 6
    cvk = (cur_tac * 6 - cv_val) * 64 + t0  # meaningful only when cur_tac
    cak = sum((pk_oh[k - 1] * (k - 1) for k in range(1, 7)), Expr()) * 64 + f0
    # integer thresholds for "after the cursor": no cursor -> -1 (everything), quiet -> past the end
    cvk_after = cvk + (one - cur_tac) * (-1) + cur_quiet * 1000  # cvk if tactical, -1 none, 999 quiet
    out.update(cur_tac=cur_tac, cur_quiet=cur_quiet, cf=cf, ct=ct)

    # ---------------------------------------------------------------- victims
    ep = s2["ep_read"]
    vs = [None] * 384
    for v in range(64):
        for w in range(1, 6):
            terms = c.ge_terms(s2["atk_us"][v], 1, lits=[them(v, w)])
            if w == 1 and 40 <= v < 48:
                fl = v - 40
                pawns = Expr()
                for a in (32 + fl - 1, 32 + fl + 1):
                    if 32 <= a < 40 and abs(a % 8 - fl) == 1:
                        pawns = pawns + us(a, 1)
                terms += c.ge_terms(pawns, 1, lits=[ep[fl]])
            vs[(6 - w) * 64 + v] = c.node(terms, f"vs[{w},{v}]")
        if v >= 56:
            vs[v] = c.node(c.t_and([empty[v], us(v - 8, 1)]), f"vs[6,{v}]")
    vs_after = [None if e is None else c.node(c.ge_terms(j - 1 - cvk_after, 0, lits=[e]), f"vsa[{j}]")
                for j, e in enumerate(vs)]
    vfirst = priority(c, vs_after, "vfirst")
    vsq_cap = [sum((vfirst[(6 - w) * 64 + v] for w in range(1, 6)), Expr()) for v in range(64)]
    vsq_pro = [vfirst[v] if v >= 56 else Expr() for v in range(64)]
    vsq = [vsq_cap[v] + vsq_pro[v] for v in range(64)]

    def attackers(cap_sel, pro_sel, tag):
        """AT[(k-1)*64 + a]: our piece of kind k on a can make the tactical move onto the selected
        victim (cap_sel: a capture / en passant square; pro_sel: a promotion push square)."""
        terms = [[] for _ in range(384)]
        for k, a, t, between in KPAIRS:
            terms[(k - 1) * 64 + a] += c.t_and([us(a, k), cap_sel[t]] + [empty[b] for b in between])
        for a in range(8, 56):
            r, f = divmod(a, 8)
            for df in (-1, 1):
                t = sq_at(r + 1, f + df)
                if t is not None:
                    terms[a] += c.t_and([us(a, 1), cap_sel[t]])
            if a + 8 >= 56:
                terms[a] += c.t_and([us(a, 1), pro_sel[a + 8]])
        return [c.node(t, f"{tag}[{j}]") if t else None for j, t in enumerate(terms)]

    at1 = attackers(vsq_cap, vsq_pro, "at1")
    afirst1 = priority(c, at1, "afirst1")
    # (a) the cursor's own victim: its next attacker after the cursor's attacker
    ct_cap = [c.node(c.t_and([ct[v], cur_tac, one - cur_promo]), f"ct.cap[{v}]") for v in range(64)]
    ct_pro = [c.node(c.t_and([ct[v], cur_promo]), f"ct.pro[{v}]") if v >= 56 else Expr() for v in range(64)]
    at0 = attackers(ct_cap, ct_pro, "at0")
    at0_after = [None if e is None else c.node(c.ge_terms(j - 1 - cak, 0, lits=[e]), f"at0a[{j}]")
                 for j, e in enumerate(at0)]
    afirst0 = priority(c, at0_after, "afirst0")

    # ---------------------------------------------------------------- quiet moves
    def quiet_terms(gate_from, tag_unused=None):
        """per (from, to): unit terms for a quiet move from a square gated by gate_from[a]."""
        terms = {}
        for k, a, t, between in KPAIRS:
            terms.setdefault((a, t), []).extend(
                c.t_and([us(a, k), empty[t], gate_from[a]] + [empty[b] for b in between]))
        for a in range(8, 48):  # pushes that are not promotions
            terms.setdefault((a, a + 8), []).extend(c.t_and([us(a, 1), empty[a + 8], gate_from[a]]))
        for a in range(8, 16):
            terms.setdefault((a, a + 16), []).extend(
                c.t_and([us(a, 1), empty[a + 8], empty[a + 16], gate_from[a]]))
        rr = s2["rel_right"]
        atk_t = s2["atk_them"]
        # castling: the rights, the rook, empty squares, king and transit squares not attacked
        ks = c.ge_terms(-(atk_t[4] + atk_t[5] + atk_t[6]), 0,
                        lits=[rr[0], us(4, 6), us(7, 4), empty[5], empty[6], gate_from[4]])
        qs = c.ge_terms(-(atk_t[4] + atk_t[3] + atk_t[2]), 0,
                        lits=[rr[1], us(4, 6), us(0, 4), empty[1], empty[2], empty[3], gate_from[4]])
        terms.setdefault((4, 6), []).extend(ks)
        terms.setdefault((4, 2), []).extend(qs)
        return terms

    ones = [one] * 64
    allq = quiet_terms(ones)
    hasq = [c.node([u for (a, tt), us_ in allq.items() if a == f for u in us_], f"hasq[{f}]") for f in range(64)]
    # (d) the next from-square with a quiet move: after f0 for a quiet cursor, else from the start
    f_after = c.node([(1, cur_quiet, f0 + 1)], "f.after") - one  # f0 if quiet, else -1
    gtf = [c.node(c.ge_terms(f - 1 - f_after, 0), f"gtf[{f}]") for f in range(64)]
    hq_after = [c.node(c.ge_terms(hasq[f], 1, lits=[gtf[f]]), f"hqa[{f}]") for f in range(64)]
    ffirst = priority(c, hq_after, "ffirst")

    def targets(sel_from, tag):
        """QT[t]: a quiet move from the selected square to t (0/1)."""
        tq = quiet_terms(sel_from)
        return [c.node([u for (a, tt), us_ in tq.items() if tt == t for u in us_], f"{tag}[{t}]")
                for t in range(64)]

    qt1 = targets(ffirst, "qt1")
    tfirst1 = priority(c, qt1, "tfirst1")
    # (c) the cursor's own from-square: the next target after t0
    cfq = [c.node(c.t_and([cf[a], cur_quiet]), f"cf.q[{a}]") for a in range(64)]
    qt0 = targets(cfq, "qt0")
    gtt = [c.node(c.ge_terms(t - 1 - t0, 0), f"gtt[{t}]") for t in range(64)]
    qt0_after = [c.node(c.t_and([qt0[t], gtt[t]]), f"qt0a[{t}]") for t in range(64)]
    tfirst0 = priority(c, qt0_after, "tfirst0")

    # ---------------------------------------------------------------- the choice: a, b, c, d in order
    full = c.node(c.ge_terms(s3["depth"], 1), "full")  # quiet moves only at full-width nodes
    ex_a = sum((e for e in afirst0), Expr())
    ex_b = sum((e for e in afirst1), Expr())
    ex_c = sum((e for e in tfirst0), Expr())
    ex_d = sum((e for e in tfirst1), Expr())
    from_a = [sum((afirst0[(k - 1) * 64 + a] for k in range(1, 7)), Expr()) for a in range(64)]
    from_b = [sum((afirst1[(k - 1) * 64 + a] for k in range(1, 7)), Expr()) for a in range(64)]
    gate = [[], [one - ex_a], [one - ex_a, one - ex_b, full], [one - ex_a, one - ex_b, one - ex_c, full]]
    srcs_from = [from_a, from_b, cf, ffirst]
    srcs_to = [ct, vsq, tfirst0, tfirst1]
    exists = [ex_a, ex_b, ex_c, ex_d]
    sel_from = [c.node([u for i in range(4) for u in c.t_and([srcs_from[i][sq], exists[i]] + gate[i])],
                       f"sel.f[{sq}]") for sq in range(64)]
    sel_to = [c.node([u for i in range(4) for u in c.t_and([srcs_to[i][sq], exists[i]] + gate[i])],
                     f"sel.t[{sq}]") for sq in range(64)]
    out.update(sel_from=sel_from, sel_to=sel_to, full=full)
    return out


# ====================================================================== stage 5: the next token

ANCHOR_AT = 25  # = machine/ref2.ANCHOR_AT


def build_stage5(c: ChessCirc, s1: dict, s2: dict, s3: dict, s4: dict) -> dict:
    from ref import MATE
    one, is_ = c.one, c.is_
    P, p_oh = s1["P"], s1["p_oh"]
    fresh = is_["START"] + c.isMOVE
    cont = c.isRET + c.isPLY
    rel = s2["rel"]
    stm_white = s2["stm_white"]
    stm_black = one - stm_white
    sel_from, sel_to = s4["sel_from"], s4["sel_to"]
    out = {}

    # ---------------------------------------------------------------- the node's own best (offset)
    own_best = c.node([(1, s3["best"][r], p_oh[r]) for r in range(MAX_PLY)], "own.best")
    own_stand = c.node([(1, s3["stand"][r], p_oh[r]) for r in range(MAX_PLY)], "own.stand")
    be_x = c.node([(1, one, own_best), (1, one, own_stand - own_best)], "be.x")
    s_feat = c.feat("ret.s", lambda t: V.decode(t)[1] if V.decode(t)[0] == "RET" else 0)
    v_x = OFF - s_feat
    improved = c.node(c.ge_terms(v_x - be_x, 1, lits=[is_["RET"]]), "improved")
    be2_x = be_x + c.node([(1, is_["RET"], v_x - be_x)], "be.raise")
    beta_x = s3["beta"] + OFF

    # ---------------------------------------------------------------- the cases
    illegal_f = c.node(c.t_and([fresh, one - p_oh[0], s2["illegal"]]), "case.illegal")
    cap_f = c.node(c.t_and([fresh, p_oh[MAX_PLY - 1], one - s2["illegal"]]), "case.plycap")
    stand_cut = c.node(c.ge_terms(s3["eval"] - s3["beta"], 0, lits=[s3["qnode"]]), "case.standcut")
    cut = c.node(c.ge_terms(be2_x - beta_x, 0, lits=[cont]), "case.cut")
    has_move = sum(sel_from, Expr())
    early = illegal_f + cap_f + stand_cut + cut
    move = c.node(c.t_and([one - early, has_move]), "case.move")
    nomove = c.node(c.t_and([one - early, one - has_move]), "case.nomove")
    done = c.node(c.t_and([nomove, p_oh[0]]), "case.done")
    mate = c.node(c.ge_terms(-be2_x, 0, lits=[s4["full"], nomove]), "case.mate")  # no legal move
    anc = c.node(c.ge_terms(s1["tm"][ANCHOR_AT - 1] + s1["tr"][ANCHOR_AT - 1], 1,
                            lits=[c.isRET, one - done]), "case.anchor")

    # ---------------------------------------------------------------- RET score
    retv = [(1, s3["eval"], cap_f), (1, s3["eval"], stand_cut), (1, be2_x - OFF, cut),
            (1, be2_x - OFF, nomove - mate - done),
            (1, P - MATE, mate + s2["in_check"] - 1)]  # mate: -MATE + P; stalemate: 0
    ret_score = c.node(retv, "ret.score")

    # ---------------------------------------------------------------- kinds
    k = {}
    k["RET_ILLEGAL"] = illegal_f
    k["RET"] = c.node(c.t_and([cap_f + stand_cut + cut + nomove - done, one - anc]), "k.RET")
    k["MOVE"] = c.node(c.t_and([move, one - improved, one - anc]), "k.MOVE")
    k["MOVE_IMP"] = c.node(c.t_and([move, improved, one - anc]), "k.MOVE_IMP")
    k["PLY"] = c.node(c.t_and([anc, one - improved]), "k.PLY")
    k["PLY_IMP"] = c.node(c.t_and([anc, improved]), "k.PLY_IMP")
    k["DONE"] = done

    # ---------------------------------------------------------------- MOVE payload (absolute)
    abs_f = [c.node(c.t_and([sel_from[s], stm_white]) + c.t_and([sel_from[s ^ 56], stm_black]), f"out.f[{s}]")
             for s in range(64)]
    abs_t = [c.node(c.t_and([sel_to[s], stm_white]) + c.t_and([sel_to[s ^ 56], stm_black]), f"out.t[{s}]")
             for s in range(64)]
    pk = [c.node([u for s in range(64) for u in c.t_and([sel_from[s], rel[s][kk - 1]])], f"out.pk=={kk}")
          for kk in range(1, 7)]
    xk = [c.node([u for s in range(64) for u in c.t_and([sel_to[s], rel[s][kk + 5]])], f"out.xk=={kk}")
          for kk in range(1, 6)]
    ep = s2["ep_read"]
    pawn_from = [rel[s][0] for s in range(64)]
    xk_ep = c.node([u for fl in range(8) for s in range(32, 40) if abs(s % 8 - fl) == 1
                    for u in c.t_and([sel_to[40 + fl], ep[fl], sel_from[s], pawn_from[s]])], "out.xk==ep")
    out.update(kinds=k, abs_f=abs_f, abs_t=abs_t, pk=pk, xk=xk + [xk_ep], white=stm_white,
               ret_score=ret_score, P=P, improved=improved)
    return out


# ====================================================================== stage 6: the bridge to Qwen

FAMS = ["START", "MOVE", "MOVE_IMP", "RET", "RET_ILLEGAL", "PLY", "PLY_IMP", "DONE", "PUT", "Q"]
FAM_ID = {n: i for i, n in enumerate(FAMS)}
NFAM = 4
NPAY = 19
CODE_AMP = 128  # a token's own code in its (tied) embedding row: CODE_AMP * (2 bit - 1)
KAPPA = 2.0**-20  # the predicted code after the exit: KAPPA * (2 bit - 1)
TAU_TEXT = 2.0**-40  # tau on text tokens; -1 on chip tokens (q35.merge)
DEPTH = 4  # the search depth START carries


def payload(tok: tuple) -> int:
    """A chip token's output-code payload (any injective layout: the code is matched against
    the tied embedding, not decoded by arithmetic)."""
    k = tok[0]
    if k == "START":
        _, depth, white, castle, ep_file = tok
        return depth | white << 4 | castle << 5 | ep_file << 9
    if k in ("MOVE", "MOVE_IMP"):
        _, f, t, pk, xk, white = tok
        return f | t << 6 | pk << 12 | xk << 15 | white << 18
    if k == "RET":
        return tok[1] + OFF
    if k in ("PLY", "PLY_IMP"):
        return tok[1]
    if k == "PUT":
        return tok[1] | tok[2] << 6
    return 0


def code_bits(fam: str, p: int) -> list:
    fid = FAM_ID[fam]
    return [fid >> i & 1 for i in range(NFAM)] + [p >> i & 1 for i in range(NPAY)]


def token_code(t: int, text) -> list | None:
    """The output-code bits of a token: a chip token's, or a printable text token's (Q)."""
    if t >= V.TEXT0:
        p = text.print_index.get(t - V.TEXT0) if text is not None else None
        return None if p is None else code_bits("Q", p)
    d = V.decode(t)
    return code_bits(d[0], payload(d))


def decode_code(bits: list, text) -> int:
    """The token with these output-code bits, or -1."""
    fid = sum(b << i for i, b in enumerate(bits[:NFAM]))
    p = sum(b << i for i, b in enumerate(bits[NFAM:]))
    if fid >= len(FAMS):
        return -1
    k = FAMS[fid]
    try:
        if k == "Q":
            q = text.print_ids.get(p)
            return -1 if q is None else V.TEXT0 + q
        if k == "START":
            return V.encode(("START", p & 15, p >> 4 & 1, p >> 5 & 15, p >> 9 & 15))
        if k in ("MOVE", "MOVE_IMP"):
            return V.encode((k, p & 63, p >> 6 & 63, p >> 12 & 7, p >> 15 & 7, p >> 18 & 1))
        if k == "RET":
            return V.encode(("RET", p - OFF))
        if k in ("PLY", "PLY_IMP"):
            return V.encode((k, p))
        if k == "PUT":
            return V.encode(("PUT", p & 63, p >> 6))
        return V.encode((k,))
    except (KeyError, ValueError):
        return -1


def build_bridge_read(c: ChessCirc) -> dict:
    """The FEN reader: Qwen's text inside <tool_call> ... </tool_call>, one token at a time.

    A line starts after a newline. Its field = the spaces so far; it is dirty once it has a token
    a FEN cannot hold (or a non-board token in field 0). Field 0 of a clean line inside a tool
    call is the board: a square cursor (FEN order) counts the squares of the tokens so far, and
    each token writes its squares into 8 tables, one per file (a token covers at most 8
    consecutive squares, so at most one per file), one row per rank. Fields 1-3 set the side,
    castling and en passant tables (cleared at field 1)."""
    one = c.one
    out = {}
    ((armed,),) = thead(c, "armed", {0: one}, {0: one},
                        [{"data": {2: c.isTCO}, "ref": one, "beta": c.isTCO + c.isTCC}])
    c.tables["armed"] = c.nodes[-1]
    reset = c.t_reset
    x_sp = c.counter("cnt.space", reset, c.t_space)
    th_sp = c.thermo(x_sp, reset, 4, "fs")
    f0, f1, f2, f3 = one - th_sp[0], th_sp[0] - th_sp[1], th_sp[1] - th_sp[2], th_sp[2] - th_sp[3]
    dirty_tok = c.t_notfen + c.node(c.t_and([f0, c.t_notboard, c.t_fenok]), "dirty.f0")
    x_dirty = c.counter("cnt.dirty", reset, dirty_tok)
    clean = one - c.thermo(x_dirty, reset, 1, "dirty")[0]
    # the square cursor: sum over the line's tokens of their width, counted per width bit
    count = Expr()
    for b in range(4):
        xb = c.counter(f"cnt.w{b}", reset, c.t_wb[b])
        count = count + (1 << b) * sum(c.thermo(xb, reset, 64 >> b, f"w{b}"), Expr())
    width = sum((c.t_wb[b] * (1 << b) for b in range(4)), Expr())
    cur = count - width  # squares before this token
    phase = [armed, f0, clean, c.t_boardok]
    at = [c.node(c.eq_terms(cur, n, lits=phase), f"fen.at{n}") for n in range(64)]
    fen = {}
    for j in range(8):
        keys, data = {}, {}
        for r in range(8):
            n = 8 * r + j
            ks = [(at[n - k], k) for k in range(8) if n - k >= 0]
            keys[r] = c.node([u for a, k in ks for u in c.t_and([a, c.t_wgt[k]])], f"fen{j}.k{r}")
            for i in range(4):
                data[2 + 4 * r + i] = c.node([u for a, k in ks for u in c.t_and([a, c.t_code[k][i]])],
                                             f"fen{j}.r{r}b{i}")
        beta = sum(keys.values(), Expr())
        res = c.multi_read(f"fen{j}", list(range(8)), keys, data, c.isTCO, beta=beta)
        for r in range(8):
            for i in range(4):
                fen[j, r, i] = res[4 * r + i]
    # side, castling, en passant (one row each; field 1 clears / sets them)
    at1 = c.node(c.t_and([armed, clean, f1]), "fen.f1")
    (side,) = thead(c, "side", {0: one}, {0: one},
                    [{"data": {2: c.t_has["w"]}, "ref": one,
                      "beta": c.node(c.t_and([at1, c.t_has["w"] + c.t_has["b"]]), "side.w")}])
    c.tables["side"] = c.nodes[-1]
    castle = []
    for ch in "KQkq":  # absolute rights 1 WK, 2 WQ, 4 BK, 8 BQ
        (cr,) = thead(c, f"castle.{ch}", {0: one}, {0: one},
                      [{"data": {2: c.t_has[ch]}, "ref": one,
                        "beta": at1 + c.node(c.t_and([armed, clean, f2, c.t_has[ch]]), f"castle.{ch}.w")}])
        c.tables[f"castle.{ch}"] = c.nodes[-1]
        castle.append(cr[0])
    any_file = sum(c.t_file, Expr())
    (ep,) = thead(c, "fen.ep", {0: one}, {0: one},
                  [{"data": {2 + x: c.node(c.t_and([f3, c.t_file[x]]), f"fen.ep{x}") for x in range(8)},
                    "ref": one, "beta": at1 + c.node(c.t_and([armed, clean, f3, any_file]), "fen.ep.w")}])
    c.tables["fen.ep"] = c.nodes[-1]
    out.update(fen=fen, side=side[0], castle=castle, ep=ep, armed=armed, cur=cur, f=(f0, f1, f2, f3),
               clean=clean)
    return out


def build_stage6(c: ChessCirc, s1: dict, s2: dict, s3: dict, s4: dict, s5: dict, br: dict,
                 depth: int = DEPTH) -> dict:
    """PUT / START emission, the best root move (BM), its echo as text, and the exit (the output
    code, ownership and tau as in rv/core.py)."""
    one, is_ = c.one, c.is_
    k = s5["kinds"]
    out = {}

    # ---------------------------------------------------------------- the root move (BM)
    bm_on = c.node(c.t_and([s5["improved"], s1["p_oh"][0]]), "bm.w")
    cb = s4["cur_bits"]
    (bm,) = thead(c, "bm", {0: one}, {0: one},
                  [{"data": {2 + i: e for i, e in enumerate(cb["f"] + cb["t"] + cb["pk"])}, "ref": one,
                    "beta": bm_on}])
    c.tables["bm"] = c.nodes[-1]
    bf, bt, bpk = bm[0:6], bm[6:12], bm[12:15]
    promo = c.node(c.t_and([bpk[0], one - bpk[1], one - bpk[2], bt[3], bt[4], bt[5]])
                   + c.t_and([bpk[0], one - bpk[1], one - bpk[2], one - bt[3], one - bt[4], one - bt[5]]),
                   "bm.promo")

    # ---------------------------------------------------------------- the echo after DONE
    xp = c.counter("cnt.print", is_["DONE"], one - is_["DONE"])
    jt = c.thermo(xp, is_["DONE"], 10, "pj")
    jeq = {v: jt[v - 1] - jt[v] for v in range(1, 10)}

    # ---------------------------------------------------------------- cases: (gate, lits, family, payload bits)
    # gate: an exact 0/1 token-feature form that multiplies the case (operands that are not
    # 0/1 on other tokens cannot fire it); lits: 0/1 conditions
    is_search = is_["START"] + c.isMOVE + c.isRET + c.isPLY
    cases = []

    def bits_of_onehot(oh, n, base=0):
        return [sum((e for v, e in enumerate(oh) if (v + base) >> i & 1), Expr()) for i in range(n)]

    # PUT: at </tool_call> and at PUT(s < 63)
    fen = br["fen"]
    put_code = []
    for i in range(4):
        terms = []
        for s in range(64):
            j, r = s % 8, 7 - s // 8
            terms.append((1, c.put_next[s], fen[j, r, i]))
        put_code.append(c.node(terms, f"put.code.b{i}"))
    put_go = c.isTCC + c.isPUT - c.put_last
    cases.append((put_go, [], "PUT", bits_of_onehot(c.put_next, 6) + put_code))
    # START after PUT(63)
    ep = br["ep"]
    ep_none = one - sum(ep, Expr())
    st_pay = [one if depth >> i & 1 else Expr() for i in range(4)] + [br["side"]] + br["castle"] \
        + bits_of_onehot(ep, 3) + [ep_none]
    cases.append((c.put_last, [], "START", st_pay))
    # the search
    F = bits_of_onehot(s5["abs_f"], 6)
    Tt = bits_of_onehot(s5["abs_t"], 6)
    pk = bits_of_onehot(s5["pk"], 3, base=1)
    xk = bits_of_onehot(s5["xk"], 3, base=1)
    mv_pay = F + Tt + pk + xk + [s5["white"]]
    for nm in ("MOVE", "MOVE_IMP"):
        cases.append((is_search, [k[nm]], nm, mv_pay))
    ret_bits = int_bits16(c, s5["ret_score"] + OFF, "ret")
    cases.append((is_search, [k["RET"]], "RET", ret_bits))
    cases.append((is_search, [k["RET_ILLEGAL"]], "RET_ILLEGAL", []))
    p_bits = bits_of_onehot(s1["p_oh"], 5)
    for nm in ("PLY", "PLY_IMP"):
        cases.append((is_search, [k[nm]], nm, p_bits))
    cases.append((is_search, [k["DONE"]], "DONE", []))
    # the echo: <tool_response> \n f t [q] \n </tool_response> \n (Qwen speaks after the last);
    # j = tokens since DONE
    def q(v):
        return [one if v >> i & 1 else Expr() for i in range(5)]

    cases.append((is_["DONE"], [], "Q", q(0)))
    tx = c.is_text
    cases.append((tx, [jeq[1]], "Q", q(1)))
    cases.append((tx, [jeq[2]], "Q", bf[0:3] + [one, Expr()]))
    cases.append((tx, [jeq[3]], "Q", bf[3:6] + [Expr(), one]))
    cases.append((tx, [jeq[4]], "Q", bt[0:3] + [one, Expr()]))
    cases.append((tx, [jeq[5]], "Q", bt[3:6] + [Expr(), one]))
    cases.append((tx, [jeq[6], promo], "Q", q(2)))
    cases.append((tx, [jeq[6], one - promo], "Q", q(1)))
    cases.append((tx, [jeq[7], promo], "Q", q(1)))
    cases.append((tx, [jeq[7], one - promo], "Q", q(3)))
    cases.append((tx, [jeq[8], promo], "Q", q(3)))
    cases.append((tx, [jeq[8], one - promo], "Q", q(1)))
    cases.append((tx, [jeq[9], promo], "Q", q(1)))
    owner = c.node([(1, c.isTCC + c.isPUT + is_search + is_["DONE"], one)]
                   + c.t_and([sum((jeq[v] for v in range(1, 9)), Expr())], a=tx)
                   + c.t_and([jeq[9], promo], a=tx), "owner")
    c.owner = owner

    # ---------------------------------------------------------------- the exit
    exit_min = max(n.level for n in c.nodes)
    for e in c.code:
        c.ffn([(-1, one, e), (1, one, -1 * e)], "code.clear", into=e)
    for i, e in enumerate(c.code):
        terms = c.t_and([owner], -KAPPA)
        for gate, lits, fam, pay in cases:
            if i < NFAM:
                if FAM_ID[fam] >> i & 1:
                    terms += c.t_and(lits + [owner], 2 * KAPPA, a=gate)
            else:
                b = i - NFAM
                if b < len(pay) and pay[b].terms:
                    terms += c.t_and(lits + [pay[b], owner], 2 * KAPPA, a=gate)
        c.ffn(terms, f"exit.code{i}", into=e, min_level=exit_min)
    c.keep = [next(iter(e.terms)) for e in c.code]
    c.ffn([(1, c.mu, one)] + c.t_and([owner, c.is_text], -TAU_TEXT), "exit.tau", into=c.tau,
          min_level=exit_min)
    c.keep += [next(iter(c.tau.terms)), next(iter(c.mu.terms))]
    out.update(owner=owner, cases=cases, promo=promo, bm=bm, jeq=jeq)
    return out


def build_all(text, depth: int = DEPTH) -> tuple:
    c = ChessCirc(text=text)
    s1 = build_stage1(c)
    s2 = build_stage2(c, s1)
    s3 = build_stage3(c, s1, s2)
    s4 = build_stage4(c, s1, s2, s3)
    s5 = build_stage5(c, s1, s2, s3, s4)
    br = build_bridge_read(c)
    s6 = build_stage6(c, s1, s2, s3, s4, s5, br, depth)
    return c, (s1, s2, s3, s4, s5, br, s6)
