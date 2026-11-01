"""Checks for the reference machine.
  1. legal moves (pseudo-legal + king-capture filter) == python-chess's, minus underpromotions
  2. the token machine == a plain recursive alpha-beta with the same semantics (move, score, nodes)
"""
import _paths  # noqa: F401  (circuit/ and machine/ on sys.path)
import random
import sys

import chess

import ref


def legal_set(pos):
    return {m for m in ref.gen_moves(pos) if not ref.can_capture_king(ref.make(pos, m))}


def to_real(m, white):
    f, t = m
    return (ref.real_square(f, white), ref.real_square(t, white))


def check_movegen(fens):
    bad = 0
    for fen in fens:
        pos, white = ref.from_fen(fen)
        mine = {to_real(m, white) for m in legal_set(pos)}
        b = chess.Board(fen)
        theirs = {(m.from_square, m.to_square) for m in b.legal_moves if m.promotion in (None, chess.QUEEN)}
        if mine != theirs:
            bad += 1
            if bad <= 3:
                print("MOVEGEN MISMATCH", fen, "extra", mine - theirs, "missing", theirs - mine)
    return bad


class Rec:
    """Recursive alpha-beta with the machine's exact semantics."""

    def __init__(self):
        self.nodes = 0

    def search(self, pos, alpha, beta, depth, ply):
        if ply and ref.can_capture_king(pos):
            return None  # illegal
        if ply >= ref.MAX_PLY - 1:
            return ref.evaluate(pos)
        if ref.in_check(pos):
            depth += 1
        best = -ref.INF
        if depth <= 0:
            best = ref.evaluate(pos)
            if best >= beta:
                return best
            alpha = max(alpha, best)
        cursor, legal, root_best = None, 0, None
        while (m := ref.next_move(pos, cursor, depth <= 0)) is not None:
            cursor = m
            self.nodes += 1
            r = self.search(ref.make(pos, m), -beta, -alpha, depth - 1, ply + 1)
            if r is None:
                continue
            v = -r
            legal += 1
            best = max(best, v)
            if v > alpha:
                alpha = v
                root_best = m
            if v >= beta:
                return best
        if depth > 0 and legal == 0:
            best = -ref.MATE + ply if ref.in_check(pos) else 0
        if ply == 0:
            self.root_best = root_best
        return best


def check_machine(fens, depth):
    bad = 0
    for fen in fens:
        pos, _ = ref.from_fen(fen)
        mc = ref.Machine(pos, depth)
        done = mc.run()
        rc = Rec()
        s = rc.search(pos, -ref.INF, ref.INF, depth, 0)
        got = ((done[1], done[2]), done[3], mc.nodes)
        want = (rc.root_best or (-1, -1), s, rc.nodes)
        if got != want:
            bad += 1
            print("MACHINE MISMATCH", fen, got, want)
    return bad


def positions(n, seed=1):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(rng.randrange(0, 90)):
            ms = list(b.legal_moves)
            if not ms:
                break
            b.push(rng.choice(ms))
        if not b.is_game_over():
            out.append(b.fen())
    return out


if __name__ == "__main__":
    special = [
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3",
    ]
    fens = special + positions(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
    print("movegen mismatches:", check_movegen(fens), "of", len(fens))
    print("machine mismatches (depth 2):", check_machine(fens[:80], 2), "of 80")
    print("machine mismatches (depth 3):", check_machine(fens[:25], 3), "of 25")
