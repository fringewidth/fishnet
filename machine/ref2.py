"""The search machine with the state layout the compiled layers can hold (spec v2).

Same search as ref.Machine (it must produce the same move, score and node count), but every
piece of state is a table the circuit keeps in Gated DeltaNet heads, and every write obeys the
circuit's one rule: a head's write may only use values computed below its own layer in the same
pass, so no table is ever rewritten from its own contents in one pass. Values that would need
that (the board, a node's best score, the ply) are carried by tokens instead.

Tables (rows keyed by ply unless noted):
  DIFF[r]    a move's change to the board, absolute coordinates: pieces removed and added,
             castling rights lost. Row 0 is the root board. The board at ply p is the sum of all
             rows (rows > p are kept empty). Written by the MOVE pass from the MOVE token
             (f, t, moved kind, captured kind); cleared by the RET pass.
  EP[r]      en passant target created by the move into ply r (from the MOVE token).
  CURSOR[r]  the last move node r tried (the MOVE token, written when it is emitted... read back
             by node r's later passes).
  BEST[r]    best child score of node r so far (-INF on creation). Two heads by ply parity, so a
             pass can reset the child's row and update its own. An update is carried by the
             token after the RET that improved it (MOVE_IMP / PLY_IMP), whose pass writes
             v = -s from the RET token one position back (the head's short convolution).
  STAND[r]   stand-pat score of a quiescence node (-INF for full-width), written when created.
  EXT[r]     1 if node r was check-extended (written when created, cleared by RET);
             depth(p) = D - p + sum of EXT rows.
  BM         the root's best move (written by a RET pass at the root that improves, from
             CURSOR[0], a lower table).
  ply        anchor + MOVEs since the anchor - RETs since the anchor (decay counters); a RET
             pass whose counters are high emits PLY(ply) instead, and that pass re-anchors.

Derived per pass, never stored: a node's alpha = max of best_eff over same-parity ancestors
(itself included), beta = -max of best_eff over the other parity (best_eff = max(BEST, STAND)).
"""

from dataclasses import dataclass, field

import ref
from ref import INF, MATE, MAX_PLY, Pos

ANCHOR_AT = 25
EP_CAPTURE = 6  # captured-kind code for en passant (a king is never captured)  # a RET pass with either counter >= this emits PLY first (counters stay <= 60)


def side_of(ply: int, root_white: bool) -> bool:
    """True if White moves at this ply."""
    return root_white == (ply % 2 == 0)


def to_abs(sq: int, white: bool) -> int:
    return sq if white else sq ^ 56


def abs_board(pos: Pos, white: bool) -> dict:
    """{(square, absolute piece code)}: codes 1-6 White, 7-12 Black."""
    out = {}
    for s, p in enumerate(pos.sq):
        if not p:
            continue
        white_piece = ref.ours(p) == white
        out[to_abs(s, white)] = ref.kind(p) + (0 if white_piece else 6)
    return out


def rel_pos(board: dict, castle_abs: int, ep_abs: int, white: bool) -> Pos:
    """The side to move's view of an absolute board (castle_abs: 1 WK, 2 WQ, 4 BK, 8 BQ)."""
    sq = [0] * 64
    for s, code in board.items():
        k = code if code <= 6 else code - 6
        mine = (code <= 6) == white
        sq[to_abs(s, white)] = k + (0 if mine else 6)
    castle = castle_abs if white else ((castle_abs >> 2) & 3) | ((castle_abs & 3) << 2)
    ep = -1 if ep_abs < 0 else to_abs(ep_abs, white)
    return Pos(tuple(sq), castle, ep)


def move_diff(f: int, t: int, pk: int, xk: int, white: bool) -> dict:
    """The DIFF row of a move in absolute coordinates, from the MOVE token alone.
    pk: moved kind; xk: captured kind (0 none, 1-5 a piece on t, EP_CAPTURE en passant).
    Returns removed / added {(sq, code)} and the castling rights the move loses."""
    me, them = (0, 6) if white else (6, 0)
    rem, add = {(f, pk + me)}, set()
    promo = pk == ref.P and t // 8 in (0, 7)
    add.add((t, (ref.Q if promo else pk) + me))
    if xk == EP_CAPTURE:
        rem.add((t - 8 if white else t + 8, ref.P + them))
    elif xk:
        rem.add((t, xk + them))
    home = 0 if white else 56
    if pk == ref.K and f == home + 4 and t == home + 6:
        rem.add((home + 7, ref.R + me))
        add.add((home + 5, ref.R + me))
    if pk == ref.K and f == home + 4 and t == home + 2:
        rem.add((home + 0, ref.R + me))
        add.add((home + 3, ref.R + me))
    lost = (3 if white else 12) if pk == ref.K else 0
    for sq, bit in ((7, 1), (0, 2), (63, 4), (56, 8)):
        if f == sq or t == sq:
            lost |= bit
    return {"rem": rem, "add": add, "lost": lost}


@dataclass
class Machine2:
    root: Pos
    root_white: bool
    depth: int
    diff: dict = field(default_factory=dict)
    ep: dict = field(default_factory=dict)
    cursor: dict = field(default_factory=dict)
    best: dict = field(default_factory=lambda: ({}, {}))  # by ply parity
    stand: dict = field(default_factory=dict)
    ext: dict = field(default_factory=dict)
    bm: tuple | None = None
    anchor: int = 0
    cm: int = 0  # MOVE tokens since the anchor
    cr: int = 0  # RET tokens since the anchor
    root_castle: int = 0
    nodes: int = 0
    tokens: list = field(default_factory=list)

    # ------------------------------------------------------------------ table reads

    def ply(self) -> int:
        return self.anchor + self.cm - self.cr

    def board_at(self, p: int) -> Pos:
        """Sum of DIFF rows 0..p (rows above p are empty by construction)."""
        assert all(not self.diff.get(r) for r in range(p + 1, MAX_PLY + 1)), "stale diff row"
        assert all(not self.ext.get(r) for r in range(p + 1, MAX_PLY + 1)), "stale ext row"
        board = dict(abs_board(self.root, self.root_white))
        lost = 0
        for r in range(1, p + 1):
            d = self.diff[r]
            for s, code in d["rem"]:
                assert board.get(s) == code, ("diff removes a piece that is not there", r, s, code)
                del board[s]
            for s, code in d["add"]:
                board[s] = code
            lost |= d["lost"]
        white = side_of(p, self.root_white)
        return rel_pos(board, self.root_castle & ~lost, self.ep.get(p, -1) if p else self._root_ep_abs, white)

    def best_eff(self, r: int) -> int:
        return max(self.best[r % 2].get(r, -INF), self.stand.get(r, -INF))

    def window(self, p: int, extra_best: int | None = None) -> tuple[int, int]:
        """(alpha, beta) of node p from the ancestors' best_eff; extra_best overrides node p's."""
        def be(r):
            v = self.best_eff(r)
            return max(v, extra_best) if (r == p and extra_best is not None) else v
        alpha = max([-INF] + [be(r) for r in range(p, -1, -2)])
        beta = -max([-INF] + [be(r) for r in range(p - 1, -1, -2)])
        return alpha, beta

    def depth_at(self, p: int) -> int:
        return self.depth - p + sum(self.ext.get(r, 0) for r in range(p + 1))

    # ------------------------------------------------------------------ tokens

    def move_token(self, pos: Pos, m, p: int, imp: bool) -> tuple:
        """MOVE(f, t, moved kind, captured kind) in absolute squares, with the improved flag."""
        f, t = m
        white = side_of(p, self.root_white)
        pk = ref.kind(pos.sq[f])
        cap = pos.sq[t]
        xk = ref.kind(cap) if ref.theirs(cap) else (EP_CAPTURE if pk == ref.P and t == pos.ep else 0)
        return ("MOVE_IMP" if imp else "MOVE", to_abs(f, white), to_abs(t, white), pk, xk, int(white))

    # ------------------------------------------------------------------ node logic

    def _select(self, p: int, pos: Pos, cursor_rel, alpha: int, beta: int, best_eff: int,
                imp: bool) -> tuple:
        """Next move after the cursor, or the node's return."""
        d = self.depth_at(p)
        m = ref.next_move(pos, cursor_rel, tactical_only=d <= 0)
        if m is not None:
            return self.move_token(pos, m, p, imp)
        if d > 0 and best_eff == -INF and self.stand.get(p, -INF) == -INF:
            score = -MATE + p if ref.in_check(pos) else 0
        else:
            score = best_eff
        if p == 0:
            return ("DONE",)
        return ("RET", score)

    def _cursor_rel(self, p: int):
        c = self.cursor.get(p)
        if c is None:
            return None
        white = side_of(p, self.root_white)
        return (to_abs(c[1], white), to_abs(c[2], white))

    def _fresh(self, p: int) -> tuple:
        """First action of node p, entered this pass (its DIFF / EP rows are written)."""
        pos = self.board_at(p)
        if p and ref.can_capture_king(pos):
            self.ext[p] = 0
            self.stand[p] = -INF
            return ("RET_ILLEGAL",)
        if p >= MAX_PLY - 1:
            self.ext[p] = 0
            self.stand[p] = -INF
            return ("RET", ref.evaluate(pos))
        self.ext[p] = 1 if ref.in_check(pos) else 0
        d = self.depth_at(p)
        self.stand[p] = ref.evaluate(pos) if d <= 0 else -INF
        alpha, beta = self.window(p)
        if d <= 0 and self.stand[p] >= beta:
            return ("RET", self.stand[p])
        return self._select(p, pos, None, alpha, beta, self.best_eff(p), imp=False)

    def _continue(self, p: int, v: int | None) -> tuple[tuple, bool]:
        """Node p after a child returned v (None: illegal / nothing to apply)."""
        pos = self.board_at(p)
        be = self.best_eff(p)
        alpha, beta = self.window(p)
        improved = v is not None and v > be
        if improved:
            be = v
            alpha, beta = self.window(p, extra_best=v)
            if p == 0:
                self.bm = self.cursor[0]
        if be >= beta:
            return ("RET", be), improved
        return self._select(p, pos, self._cursor_rel(p), alpha, beta, be, imp=improved), improved

    # ------------------------------------------------------------------ one pass

    def step(self, tok: tuple, prev: tuple | None) -> tuple:
        k = tok[0]
        # writes carried by this token (bottom layers, from its payload / the previous token)
        if k in ("MOVE_IMP", "PLY_IMP"):
            p = self.ply()
            self.best[p % 2][p] = -prev[1]
        if k == "START":
            self.anchor, self.cm, self.cr = 0, 0, 0
            self.best[0][0] = -INF
            out = self._fresh(0)
        elif k in ("MOVE", "MOVE_IMP"):
            p = self.ply()
            _, f, t, pk, xk, white = tok
            self.cursor[p] = tok
            self.diff[p + 1] = move_diff(f, t, pk, xk, bool(white))
            self.ep[p + 1] = (f + t) // 2 if pk == ref.P and abs(t - f) == 16 else -1
            self.best[(p + 1) % 2][p + 1] = -INF
            self.cm += 1
            self.nodes += 1
            out = self._fresh(p + 1)
        elif k in ("RET", "RET_ILLEGAL"):
            p = self.ply()  # the returning child
            self.diff[p] = {}
            self.ext[p] = 0
            self.cr += 1
            out, improved = self._continue(p - 1, -tok[1] if k == "RET" else None)
            # a RET pass with high counters re-anchors before acting; the PLY pass then acts
            if (self.cm >= ANCHOR_AT or self.cr >= ANCHOR_AT) and out[0] != "DONE":
                return ("PLY_IMP" if improved else "PLY", self.ply())
        elif k in ("PLY", "PLY_IMP"):
            self.anchor, self.cm, self.cr = tok[1], 0, 0
            out, _ = self._continue(self.ply(), None)
        else:
            raise ValueError(tok)
        return out

    def focus_abs_board(self) -> dict:
        """(test hook) the focus node's board in absolute coordinates."""
        p = self.ply()
        return abs_board(self.board_at(p), side_of(p, self.root_white))

    def start_token(self) -> tuple:
        ep = self.root.ep
        ep_file = 8 if ep < 0 else to_abs(ep, self.root_white) % 8
        return ("START", self.depth, int(self.root_white), self.root_castle, ep_file)

    def run(self, max_tokens: int = 10**8, hook=None) -> tuple:
        self._root_ep_abs = -1 if self.root.ep < 0 else to_abs(self.root.ep, self.root_white)
        c = self.root.castle
        self.root_castle = c if self.root_white else ((c >> 2) & 3) | ((c & 3) << 2)
        tok, prev = self.start_token(), None
        self.tokens = [tok]
        while tok[0] != "DONE" and len(self.tokens) < max_tokens:
            nxt = self.step(tok, prev)
            if hook:
                hook(self, tok)
            tok, prev = nxt, tok
            self.tokens.append(tok)
        return tok
