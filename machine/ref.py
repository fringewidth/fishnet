"""Reference semantics of the chess search machine: the token stream the compiled layers must emit.

The machine is a fixed-depth alpha-beta search with a captures-only quiescence search, written
as a token loop with explicit per-ply state (no recursion), the way the circuit holds it:

  state   board[p], ctrl[p] for each ply p (DeltaNet rows), the ply pointer, the root's best move
  tokens  START                 begin the search at ply 0
          MOVE(f, t)            emitted by the pass that picks a move; the pass reading it makes
                                the move (child board at ply+1) and decides the child's first action
          RET(s) / RET_ILLEGAL  emitted by a node's last pass; the pass reading it updates the
                                parent (alpha, best, cutoff) and picks the parent's next action
          DONE(f, t, s)         the root has finished

Every board is stored from the side to move's view ("us" = pieces 1..6, moving towards rank 8;
"them" = 7..12). Making a move writes the child mirrored (sq ^ 56) with colours swapped, so move
generation only ever exists for one side.

A node whose side to move is in check searches one ply deeper (check extension), so mates at
the horizon are seen; a node at the last ply row returns its static evaluation.

Move order at a node: phase C (captures and queen promotions) by most valuable victim, then
victim square, then least valuable attacker, then attacker square; phase Q (quiet moves, full-
width nodes only) by from-square, then to-square. A node resumes after its cursor (the last move
it tried). Legality is by king capture: a node whose side to move can capture the enemy king
returns RET_ILLEGAL at once. Promotions are to a queen only.
"""

from dataclasses import dataclass, field

from pesto import EG, MG

EMPTY = 0
P, N, B, R, Q, K = 1, 2, 3, 4, 5, 6
VAL = {P: 1, N: 2, B: 3, R: 4, Q: 5, K: 6}
PROMO_VICTIM = 6  # a quiet queen promotion sorts above every capture's victim value
INF = 32000
MATE = 31000
MAX_PLY = 32  # board / ctrl rows; a node at the last row returns its static eval

KNIGHT = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
KING = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
DIAG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ORTH = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def ours(p: int) -> bool:
    return 1 <= p <= 6


def theirs(p: int) -> bool:
    return 7 <= p <= 12


def kind(p: int) -> int:
    return p if p <= 6 else p - 6


def at(r: int, f: int) -> int | None:
    return r * 8 + f if 0 <= r < 8 and 0 <= f < 8 else None


@dataclass(frozen=True)
class Pos:
    sq: tuple  # 64 piece codes, side to move's view
    castle: int  # 1 our O-O, 2 our O-O-O, 4 their O-O, 8 their O-O-O
    ep: int  # en passant target square (rank 6 in our view) or -1


# ---------------------------------------------------------------------------- attacks


def attacks_from(pos: Pos, s: int) -> list[int]:
    """Squares the piece on s attacks (pawns: their capture squares, by colour)."""
    p = pos.sq[s]
    if not p:
        return []
    r, f = divmod(s, 8)
    k = kind(p)
    out = []
    if k == P:
        dr = 1 if ours(p) else -1
        for df in (-1, 1):
            t = at(r + dr, f + df)
            if t is not None:
                out.append(t)
    elif k in (N, K):
        for dr, df in KNIGHT if k == N else KING:
            t = at(r + dr, f + df)
            if t is not None:
                out.append(t)
    else:
        dirs = (DIAG if k == B else ORTH if k == R else DIAG + ORTH)
        for dr, df in dirs:
            rr, ff = r + dr, f + df
            while (t := at(rr, ff)) is not None:
                out.append(t)
                if pos.sq[t]:
                    break
                rr, ff = rr + dr, ff + df
    return out


def attacked_by(pos: Pos, s: int, us: bool) -> bool:
    side = ours if us else theirs
    return any(side(pos.sq[a]) and s in attacks_from(pos, a) for a in range(64))


def king_sq(pos: Pos, us: bool) -> int:
    want = K if us else K + 6
    return pos.sq.index(want)


def in_check(pos: Pos) -> bool:
    return attacked_by(pos, king_sq(pos, True), us=False)


def can_capture_king(pos: Pos) -> bool:
    return attacked_by(pos, king_sq(pos, False), us=True)


# ---------------------------------------------------------------------------- moves


def gen_moves(pos: Pos) -> list[tuple[int, int]]:
    """Pseudo-legal moves (f, t) for the side to move. Promotions are implied queen."""
    out = []
    for f in range(64):
        p = pos.sq[f]
        if not ours(p):
            continue
        r, fi = divmod(f, 8)
        if p == P:
            t = f + 8
            if not pos.sq[t]:
                out.append((f, t))
                if r == 1 and not pos.sq[f + 16]:
                    out.append((f, f + 16))
            for df in (-1, 1):
                t = at(r + 1, fi + df)
                if t is not None and (theirs(pos.sq[t]) or t == pos.ep):
                    out.append((f, t))
            continue
        for t in attacks_from(pos, f):
            if not ours(pos.sq[t]):
                out.append((f, t))
    if pos.sq[4] == K:
        if (pos.castle & 1 and pos.sq[7] == R and not pos.sq[5] and not pos.sq[6]
                and not any(attacked_by(pos, s, us=False) for s in (4, 5, 6))):
            out.append((4, 6))
        if (pos.castle & 2 and pos.sq[0] == R and not pos.sq[1] and not pos.sq[2] and not pos.sq[3]
                and not any(attacked_by(pos, s, us=False) for s in (4, 3, 2))):
            out.append((4, 2))
    return out


def is_tactical(pos: Pos, m) -> bool:
    """Phase C: captures (incl. en passant) and queen promotions."""
    f, t = m
    return bool(theirs(pos.sq[t]) or (pos.sq[f] == P and (t == pos.ep or t >= 56)))


def order_key(pos: Pos, m) -> tuple:
    f, t = m
    if is_tactical(pos, m):
        if theirs(pos.sq[t]):
            victim = VAL[kind(pos.sq[t])]
        elif pos.sq[f] == P and t == pos.ep:
            victim = VAL[P]
        else:
            victim = PROMO_VICTIM
        return (0, -victim, t, VAL[kind(pos.sq[f])], f)
    return (1, f, t, 0, 0)


def next_move(pos: Pos, cursor, tactical_only: bool):
    """The first move in order after the cursor (None: from the start)."""
    moves = [m for m in gen_moves(pos) if not tactical_only or is_tactical(pos, m)]
    after = None if cursor is None else order_key(pos, cursor)
    best = None
    for m in moves:
        k = order_key(pos, m)
        if after is not None and k <= after:
            continue
        if best is None or k < best[0]:
            best = (k, m)
    return None if best is None else best[1]


def make(pos: Pos, m) -> Pos:
    """The child position after move m, mirrored to the new side to move's view."""
    f, t = m
    sq = list(pos.sq)
    p = sq[f]
    sq[t], sq[f] = p, EMPTY
    if p == P and t == pos.ep:
        sq[t - 8] = EMPTY
    if p == P and t >= 56:
        sq[t] = Q
    if p == K and f == 4 and t == 6:
        sq[7], sq[5] = EMPTY, R
    if p == K and f == 4 and t == 2:
        sq[0], sq[3] = EMPTY, R
    c = pos.castle
    if p == K:
        c &= ~3
    if f == 7:
        c &= ~1
    if f == 0:
        c &= ~2
    if t == 63:
        c &= ~4
    if t == 56:
        c &= ~8
    ep = f + 8 if p == P and t - f == 16 else -1
    flip = [EMPTY] * 64
    for s in range(64):
        q = sq[s]
        flip[s ^ 56] = q + 6 if ours(q) else q - 6 if theirs(q) else EMPTY
    return Pos(tuple(flip), ((c >> 2) & 3) | ((c & 3) << 2), -1 if ep < 0 else ep ^ 56)


# ---------------------------------------------------------------------------- evaluation


def evaluate(pos: Pos) -> int:
    """PeSTO tapered evaluation, side to move's view: floor((mg*ph + eg*(24-ph)) / 24)."""
    mg = eg = ph = 0
    for s, p in enumerate(pos.sq):
        if not p:
            continue
        k = kind(p)
        ph += (0, 0, 1, 1, 2, 4, 0)[k]
        if ours(p):
            mg += MG[k - 1][s]
            eg += EG[k - 1][s]
        else:
            mg -= MG[k - 1][s ^ 56]
            eg -= EG[k - 1][s ^ 56]
    ph = min(ph, 24)
    return (mg * ph + eg * (24 - ph)) // 24


# ---------------------------------------------------------------------------- the machine


@dataclass
class Ctrl:
    alpha: int
    beta: int
    best: int
    depth: int
    cursor: tuple | None = None
    legal: int = 0


@dataclass
class Machine:
    root: Pos
    depth: int
    board: dict = field(default_factory=dict)
    ctrl: dict = field(default_factory=dict)
    ply: int = 0
    root_best: tuple | None = None
    nodes: int = 0
    tokens: list = field(default_factory=list)

    def _enter(self) -> tuple:
        """First action of the node just entered at self.ply."""
        pos, c = self.board[self.ply], self.ctrl[self.ply]
        if self.ply and can_capture_king(pos):
            return ("RET_ILLEGAL",)
        if self.ply >= MAX_PLY - 1:
            return ("RET", evaluate(pos))
        if in_check(pos):
            c.depth += 1  # check extension: a side in check never stands pat
        if c.depth <= 0:
            stand = evaluate(pos)
            c.best = stand
            if stand >= c.beta:
                return ("RET", stand)
            c.alpha = max(c.alpha, stand)
        return self._next()

    def _next(self) -> tuple:
        """The node's next move, or its return when it has none left."""
        pos, c = self.board[self.ply], self.ctrl[self.ply]
        m = next_move(pos, c.cursor, tactical_only=c.depth <= 0)
        if m is not None:
            return ("MOVE", *m)
        if c.depth > 0 and c.legal == 0:
            score = -MATE + self.ply if in_check(pos) else 0
        else:
            score = c.best
        if self.ply == 0:
            return ("DONE", *(self.root_best or (-1, -1)), score)
        return ("RET", score)

    def step(self, tok: tuple) -> tuple:
        """One pass: read the token, update state, return the next token."""
        k = tok[0]
        if k == "START":
            self.ply = 0
            self.ctrl[0] = Ctrl(-INF, INF, -INF, self.depth)
            self.board[0] = self.root
            return self._enter()
        if k == "MOVE":
            m = (tok[1], tok[2])
            parent = self.ctrl[self.ply]
            parent.cursor = m
            self.board[self.ply + 1] = make(self.board[self.ply], m)
            self.ply += 1
            self.ctrl[self.ply] = Ctrl(-parent.beta, -parent.alpha, -INF, parent.depth - 1)
            self.nodes += 1
            return self._enter()
        if k in ("RET", "RET_ILLEGAL"):
            self.ply -= 1
            c = self.ctrl[self.ply]
            if k == "RET":
                v = -tok[1]
                c.legal += 1
                if v > c.best:
                    c.best = v
                if v > c.alpha:
                    c.alpha = v
                    if self.ply == 0:
                        self.root_best = c.cursor
                if v >= c.beta:
                    return ("RET", c.best)
            return self._next()
        raise ValueError(tok)

    def run(self, max_tokens: int = 10**8) -> tuple:
        tok = ("START",)
        self.tokens = [tok]
        while tok[0] != "DONE" and len(self.tokens) < max_tokens:
            tok = self.step(tok)
            self.tokens.append(tok)
        return tok


# ---------------------------------------------------------------------------- FEN


def from_fen(fen: str) -> tuple[Pos, bool]:
    """(position in the side to move's view, True if White is to move)."""
    parts = fen.split()
    rows = parts[0].split("/")
    sq = [EMPTY] * 64
    codes = {"p": P, "n": N, "b": B, "r": R, "q": Q, "k": K}
    for i, row in enumerate(rows):
        r, f = 7 - i, 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
                continue
            white = ch.isupper()
            sq[r * 8 + f] = codes[ch.lower()] + (0 if white else 6)
            f += 1
    white_to_move = parts[1] == "w"
    cr = parts[2] if len(parts) > 2 else "-"
    castle = (1 if "K" in cr else 0) | (2 if "Q" in cr else 0) | (4 if "k" in cr else 0) | (8 if "q" in cr else 0)
    ep = -1
    if len(parts) > 3 and parts[3] != "-":
        ep = (int(parts[3][1]) - 1) * 8 + "abcdefgh".index(parts[3][0])
    if not white_to_move:
        flip = [EMPTY] * 64
        for s in range(64):
            q = sq[s]
            flip[s ^ 56] = q + 6 if ours(q) else q - 6 if theirs(q) else EMPTY
        sq = flip
        castle = ((castle >> 2) & 3) | ((castle & 3) << 2)
        ep = -1 if ep < 0 else ep ^ 56
    return Pos(tuple(sq), castle, ep), white_to_move


def real_square(s: int, white_to_move: bool) -> int:
    return s if white_to_move else s ^ 56
