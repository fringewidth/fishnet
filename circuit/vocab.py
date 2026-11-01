"""Chip token ids of the chess machine and their decoding.

Families (ids from 0, in this order):
  START(depth 1..15, root_white, castle 0..15, ep_file 0..8 [8 = none])
  MOVE(f, t, pk, xk, white) and MOVE_IMP(...)   absolute squares, mover's colour; only
                                       geometrically possible (f, t, pk)
  RET(s) for s in [-INF, INF], RET_ILLEGAL
  PLY(p) and PLY_IMP(p) for p in 0..31
  DONE
  PUT(s, code 0..12)                   the bridge loads the root board, one square per token
Text tokens (Qwen's vocabulary) are TEXT0 + qwen id.
"""
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/machine")
import ref  # noqa: E402
from ref import INF, MAX_PLY  # noqa: E402

EP_CAPTURE = 6


def _geometric_moves():
    """Every (f, t, pk) a pseudo-legal move of either colour can have (absolute squares)."""
    out = set()
    for f in range(64):
        r, fi = divmod(f, 8)
        for pk in (ref.N, ref.B, ref.R, ref.Q, ref.K):
            dirs = {ref.N: ref.KNIGHT, ref.K: ref.KING, ref.B: ref.DIAG, ref.R: ref.ORTH,
                    ref.Q: ref.DIAG + ref.ORTH}[pk]
            reach = 1 if pk in (ref.N, ref.K) else 7
            for dr, df in dirs:
                for k in range(1, reach + 1):
                    t = ref.at(r + dr * k, fi + df * k)
                    if t is None:
                        break
                    out.add((f, t, pk))
        for dr, start in ((1, 1), (-1, 6)):  # white pawns move up from rank 1, black down from 6
            if r in (0, 7):
                continue
            t = ref.at(r + dr, fi)
            out.add((f, t, ref.P))
            if r == start:
                out.add((f, ref.at(r + 2 * dr, fi), ref.P))
            for df in (-1, 1):
                t = ref.at(r + dr, fi + df)
                if t is not None:
                    out.add((f, t, ref.P))
    for f, t in ((4, 6), (4, 2), (60, 62), (60, 58)):
        out.add((f, t, ref.K))
    return sorted(out)


GEO = _geometric_moves()
MOVES = [(f, t, pk, xk, w) for f, t, pk in GEO for xk in range(7) for w in (0, 1)
         if (xk != EP_CAPTURE or (pk == ref.P and f % 8 != t % 8))
         and (pk != ref.P or (t > f) == bool(w))]  # pawns move up for White, down for Black

FAMILIES = []
_base = 0


def _family(name, size):
    global _base
    FAMILIES.append((name, _base, size))
    _base += size
    return _base - size


START0 = _family("START", 16 * 2 * 16 * 9)
MOVE0 = _family("MOVE", len(MOVES))
MOVE_IMP0 = _family("MOVE_IMP", len(MOVES))
RET0 = _family("RET", 2 * INF + 1)
RET_ILLEGAL = _family("RET_ILLEGAL", 1)
PLY0 = _family("PLY", MAX_PLY)
PLY_IMP0 = _family("PLY_IMP", MAX_PLY)
DONE = _family("DONE", 1)
PUT0 = _family("PUT", 64 * 13)
VOCAB = _base
TEXT0 = 1 << 20
MOVE_INDEX = {m: i for i, m in enumerate(MOVES)}


def encode(tok: tuple) -> int:
    k = tok[0]
    if k == "START":
        _, depth, white, castle, ep_file = tok
        return START0 + ((depth * 2 + white) * 16 + castle) * 9 + ep_file
    if k in ("MOVE", "MOVE_IMP"):
        return (MOVE0 if k == "MOVE" else MOVE_IMP0) + MOVE_INDEX[tok[1:6]]
    if k == "RET":
        return RET0 + tok[1] + INF
    if k == "RET_ILLEGAL":
        return RET_ILLEGAL
    if k in ("PLY", "PLY_IMP"):
        return (PLY0 if k == "PLY" else PLY_IMP0) + tok[1]
    if k == "DONE":
        return DONE
    if k == "PUT":
        return PUT0 + tok[1] * 13 + tok[2]
    if k == "TEXT":
        return TEXT0 + tok[1]
    raise ValueError(tok)


def decode(i: int) -> tuple:
    if i >= TEXT0:
        return ("TEXT", i - TEXT0)
    for name, base, size in FAMILIES:
        if base <= i < base + size:
            j = i - base
            if name == "START":
                ep_file, j = j % 9, j // 9
                castle, j = j % 16, j // 16
                return ("START", j // 2, j % 2, castle, ep_file)
            if name in ("MOVE", "MOVE_IMP"):
                return (name, *MOVES[j])
            if name == "RET":
                return ("RET", j - INF)
            if name in ("PLY", "PLY_IMP"):
                return (name, j)
            if name == "PUT":
                return ("PUT", j // 13, j % 13)
            return (name,)
    raise ValueError(i)


def label(tok: tuple) -> str:
    """A short readable name of a decoded chip token (for the web page and the video)."""
    def sq(s):
        return "abcdefgh"[s % 8] + str(s // 8 + 1)
    k = tok[0]
    if k == "PUT":
        return f"PUT {sq(tok[1])} {' PNBRQKpnbrqk'[tok[2]].strip() or '.'}"
    if k == "START":
        return f"START depth {tok[1]}"
    if k in ("MOVE", "MOVE_IMP"):
        piece = " PNBRQK"[tok[3]] if tok[5] else " pnbrqk"[tok[3]]
        cap = "x" if tok[4] else "-"
        return f"{'MOVE*' if k == 'MOVE_IMP' else 'MOVE'} {piece}{sq(tok[1])}{cap}{sq(tok[2])}"
    if k == "RET":
        return f"RET {tok[1]:+d}"
    if k == "RET_ILLEGAL":
        return "RET illegal"
    if k in ("PLY", "PLY_IMP"):
        return f"{k} {tok[1]}"
    return k


if __name__ == "__main__":
    print("geometric (f,t,pk):", len(GEO), " move tokens:", len(MOVES), " vocab:", VOCAB)
    for t in [("START", 5, 1, 15, 8), ("MOVE", 12, 28, 1, 0, 1), ("MOVE_IMP", 4, 6, 6, 0, 1), ("RET", -31000),
              ("RET_ILLEGAL",), ("PLY_IMP", 7), ("DONE",),
              ("PUT", 63, 12), ("PUT", 5, 0), ("TEXT", 248000)]:
        assert decode(encode(t)) == t, t
    print("round trip ok")
