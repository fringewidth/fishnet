"""Run the circuit (the repo's exact IR evaluator, fp32) over spec v2's token stream and compare
its dims with the spec at every token."""
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/machine")
import build as B  # noqa: E402  (imports fp32 first)
import ref  # noqa: E402
import ref2  # noqa: E402
import vocab as V  # noqa: E402
from chipformer.q35 import ir  # noqa: E402
from chipformer.q35.ir import A_BIT, A_REF, REF_COLS  # noqa: E402


def initial_carry(c, root_abs: dict) -> dict:
    """Initial head states: every summed table's ballast row is set; the root-board tables hold
    the root board (as the bridge's TCC and PUT tokens leave them)."""
    carry = {}
    for ni, node in enumerate(c.nodes):
        if node.name not in c.multi:
            continue
        bal = np.zeros(node.n_cols)
        bal[list(REF_COLS)] = A_REF
        rows = {B.BAL: bal}
        if node.name.startswith("rb"):
            bit = int(node.name[2:])
            for r in c.multi[node.name]:
                row = np.zeros(node.n_cols)
                if root_abs.get(r, 0) >> bit & 1:
                    row[2 + r] = A_BIT
                rows[r] = row
        carry[ni, "S", 0] = rows
    return carry


def spec_trace(fen: str, depth: int):
    pos, white = ref.from_fen(fen)
    m = ref2.Machine2(pos, white, depth)
    seen = []
    m.run(hook=lambda mm, tok: seen.append((mm.ply(), mm.focus_abs_board())))
    root_abs = ref2.abs_board(pos, white)
    return m, seen, root_abs


def check_stage1(fen: str, depth: int) -> int:
    m, seen, root_abs = spec_trace(fen, depth)
    c = B.ChessCirc()
    dims = B.build_stage1(c)
    toks = [V.encode(t) for t in m.tokens[:-1]]  # every token that a pass reads
    t0 = time.time()
    mat = ir.evaluate(c, toks, check=True, carry=initial_carry(c, root_abs))
    dt = time.time() - t0
    P = ir.ev(dims["P"], mat)
    board = np.stack([np.stack([ir.ev(dims["board"][s][k - 1], mat) for k in range(1, 13)], 1)
                      for s in range(64)], 1)  # (n, 64, 12)
    bad = 0
    for i, (ply, want) in enumerate(seen):
        exp = np.zeros((64, 12))
        for s, code in want.items():
            exp[s, code - 1] = 1
        if P[i] != ply or not np.array_equal(board[i], exp):
            bad += 1
            if bad <= 3:
                diffsq = [(s, int(np.argmax(board[i][s])) + 1 if board[i][s].any() else 0, want.get(s, 0))
                          for s in range(64) if not np.array_equal(board[i][s], exp[s])]
                print(f"  token {i} {m.tokens[i]}: P {P[i]} want {ply}; squares (got, want) {diffsq[:6]}")
    print(f"stage1 {fen[:40]}.. d{depth}: {len(seen)} tokens, mismatches {bad}, "
          f"circuit {c.stats()}, eval {dt:.1f}s")
    return bad


def check_stage2(fen: str, depth: int, sample: int = 37) -> int:
    pos, white = ref.from_fen(fen)
    m = ref2.Machine2(pos, white, depth)
    seen = []
    m.run(hook=lambda mm, tok: seen.append(mm.board_at(mm.ply())))
    c = B.ChessCirc()
    s1 = B.build_stage1(c)
    s2 = B.build_stage2(c, s1)
    toks = [V.encode(t) for t in m.tokens[:-1]]
    mat = ir.evaluate(c, toks, check=True, carry=initial_carry(c, ref2.abs_board(pos, white)))
    ev = lambda e: ir.ev(e, mat)  # noqa: E731
    rel = np.stack([np.stack([ev(s2["rel"][s][k]) for k in range(12)], 1) for s in range(64)], 1)
    ill, chk = ev(s2["illegal"]), ev(s2["in_check"])
    rights = np.stack([ev(e) for e in s2["rel_right"]], 1)
    epr = np.stack([ev(e) for e in s2["ep_read"]], 1)
    atk_us = np.stack([ev(e) for e in s2["atk_us"]], 1)
    atk_them = np.stack([ev(e) for e in s2["atk_them"]], 1)
    bad = 0
    for i, pos_i in enumerate(seen):
        exp = np.zeros((64, 12))
        for s, p in enumerate(pos_i.sq):
            if p:
                exp[s, p - 1] = 1
        want_r = np.array([pos_i.castle >> b & 1 for b in range(4)])
        want_ep = np.zeros(8)
        if pos_i.ep >= 0:
            want_ep[pos_i.ep % 8] = 1
        errs = []
        if not np.array_equal(rel[i], exp):
            errs.append("rel board")
        if (ill[i] >= 1) != ref.can_capture_king(pos_i):
            errs.append(f"illegal {ill[i]}")
        if (chk[i] >= 1) != ref.in_check(pos_i):
            errs.append(f"in_check {chk[i]}")
        if not np.array_equal(rights[i], want_r):
            errs.append(f"rights {rights[i]} want {want_r}")
        if not np.array_equal(epr[i], want_ep):
            errs.append(f"ep {epr[i]} want {want_ep}")
        if i % sample == 0:
            for s in range(64):
                if (atk_us[i, s] >= 1) != ref.attacked_by(pos_i, s, True) or \
                   (atk_them[i, s] >= 1) != ref.attacked_by(pos_i, s, False):
                    errs.append(f"attack map at {s}")
                    break
        if errs:
            bad += 1
            if bad <= 3:
                print(f"  token {i} {m.tokens[i]}: {errs}")
    print(f"stage2 {fen[:40]}.. d{depth}: {len(seen)} tokens, mismatches {bad}, circuit {c.stats()}")
    return bad


def check_stage3(fen: str, depth: int, sample: int = 1) -> int:
    pos, white = ref.from_fen(fen)
    m = ref2.Machine2(pos, white, depth)
    seen = []

    def hook(mm, tok):
        p = mm.ply()
        a, b = mm.window(p)
        seen.append((p, mm.board_at(p), mm.depth_at(p), mm.stand.get(p, -ref.INF), a, b))

    m.run(hook=hook)
    c = B.ChessCirc()
    s1 = B.build_stage1(c)
    s2 = B.build_stage2(c, s1)
    s3 = B.build_stage3(c, s1, s2)
    toks = [V.encode(t) for t in m.tokens[:-1]]
    t0 = time.time()
    mat = ir.evaluate(c, toks, check=True, carry=initial_carry(c, ref2.abs_board(pos, white)))
    dt = time.time() - t0
    ev = lambda e: ir.ev(e, mat)  # noqa: E731
    E, D, ST, AL, BE = ev(s3["eval"]), ev(s3["depth"]), ev(s3["stand_x"]), ev(s3["alpha"]), ev(s3["beta"])

    def inf(v):  # the circuit's -INF is -32768, the spec's -32000
        return -ref.INF if v <= -ref.INF else (ref.INF if v >= ref.INF else v)

    bad = 0
    for i, (p, pos_i, d, st, a, b) in enumerate(seen):
        errs = []
        if E[i] != ref.evaluate(pos_i):
            errs.append(f"eval {E[i]} want {ref.evaluate(pos_i)}")
        if D[i] != d:
            errs.append(f"depth {D[i]} want {d}")
        tok = m.tokens[i][0]
        if tok in ("START", "MOVE", "MOVE_IMP"):
            got_st = ST[i] - B.OFF if ST[i] else -ref.INF
            if got_st != st:
                errs.append(f"stand {got_st} want {st}")
        if inf(AL[i]) != a or inf(BE[i]) != b:
            errs.append(f"window ({AL[i]}, {BE[i]}) want ({a}, {b})")
        if errs:
            bad += 1
            if bad <= 4:
                print(f"  token {i} {m.tokens[i]} P={p}: {errs}")
    print(f"stage3 {fen[:40]}.. d{depth}: {len(seen)} tokens, mismatches {bad}, circuit {c.stats()}, eval {dt:.0f}s")
    return bad


def check_stage4(fen: str, depth: int) -> int:
    pos, white = ref.from_fen(fen)
    m = ref2.Machine2(pos, white, depth)
    seen = []

    def hook(mm, tok):
        p = mm.ply()
        b = mm.board_at(p)
        fresh = tok[0] in ("START", "MOVE", "MOVE_IMP")
        cur = None if fresh else mm._cursor_rel(p)
        skip = p > 0 and ref.can_capture_king(b)
        want = None if skip else ref.next_move(b, cur, tactical_only=mm.depth_at(p) <= 0)
        seen.append((skip, want))

    m.run(hook=hook)
    c = B.ChessCirc()
    s1 = B.build_stage1(c)
    s2 = B.build_stage2(c, s1)
    s3 = B.build_stage3(c, s1, s2)
    s4 = B.build_stage4(c, s1, s2, s3)
    toks = [V.encode(t) for t in m.tokens[:-1]]
    t0 = time.time()
    mat = ir.evaluate(c, toks, check=True, carry=initial_carry(c, ref2.abs_board(pos, white)))
    dt = time.time() - t0
    SF = np.stack([ir.ev(e, mat) for e in s4["sel_from"]], 1)
    ST = np.stack([ir.ev(e, mat) for e in s4["sel_to"]], 1)
    bad = checked = 0
    for i, (skip, want) in enumerate(seen):
        if skip:
            continue
        checked += 1
        got = None
        if SF[i].sum() == 1 and ST[i].sum() == 1:
            got = (int(SF[i].argmax()), int(ST[i].argmax()))
        elif SF[i].sum() or ST[i].sum():
            got = ("bad one-hot", SF[i].sum(), ST[i].sum())
        if got != want:
            bad += 1
            if bad <= 5:
                print(f"  token {i} {m.tokens[i]}: got {got} want {want}")
    print(f"stage4 {fen[:40]}.. d{depth}: {checked} checked, mismatches {bad}, circuit {c.stats()}, eval {dt:.0f}s")
    return bad


def predicted_tokens(c, s5, mat):
    """The token the circuit predicts at every position, decoded from its output dims."""
    ev = lambda e: ir.ev(e, mat)  # noqa: E731
    kinds = {n: ev(e) for n, e in s5["kinds"].items()}
    F = np.stack([ev(e) for e in s5["abs_f"]], 1)
    T = np.stack([ev(e) for e in s5["abs_t"]], 1)
    PK = np.stack([ev(e) for e in s5["pk"]], 1)
    XK = np.stack([ev(e) for e in s5["xk"]], 1)
    W, S, P = ev(s5["white"]), ev(s5["ret_score"]), ev(s5["P"])
    out = []
    for i in range(mat.shape[0]):
        on = [n for n, v in kinds.items() if v[i] == 1]
        bad = [n for n, v in kinds.items() if v[i] not in (0, 1)]
        if len(on) != 1 or bad:
            out.append(("?", on, bad))
            continue
        n = on[0]
        if n in ("MOVE", "MOVE_IMP"):
            xk = 0 if XK[i].sum() == 0 else int(XK[i].argmax()) + 1
            out.append((n, int(F[i].argmax()), int(T[i].argmax()), int(PK[i].argmax()) + 1, xk, int(W[i])))
        elif n == "RET":
            out.append((n, int(S[i])))
        elif n in ("PLY", "PLY_IMP"):
            out.append((n, int(P[i])))
        else:
            out.append((n,))
    return out


def check_stage5(fen: str, depth: int) -> int:
    pos, white = ref.from_fen(fen)
    m = ref2.Machine2(pos, white, depth)
    m.run()
    c = B.ChessCirc()
    s1 = B.build_stage1(c)
    s2 = B.build_stage2(c, s1)
    s3 = B.build_stage3(c, s1, s2)
    s4 = B.build_stage4(c, s1, s2, s3)
    s5 = B.build_stage5(c, s1, s2, s3, s4)
    toks = [V.encode(t) for t in m.tokens[:-1]]
    t0 = time.time()
    mat = ir.evaluate(c, toks, check=True, carry=initial_carry(c, ref2.abs_board(pos, white)))
    dt = time.time() - t0
    pred = predicted_tokens(c, s5, mat)
    bad = 0
    for i, want in enumerate(m.tokens[1:]):
        if pred[i] != want:
            bad += 1
            if bad <= 6:
                print(f"  after token {i} {m.tokens[i]}: predicted {pred[i]} want {want}")
    print(f"stage5 {fen[:40]}.. d{depth}: {len(pred)} next-token predictions, mismatches {bad}, "
          f"circuit {c.stats()}, eval {dt:.0f}s")
    return bad


if __name__ == "__main__":
    fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"]
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    stage = sys.argv[2] if len(sys.argv) > 2 else "1"
    check = {"1": check_stage1, "2": check_stage2, "3": check_stage3, "4": check_stage4, "5": check_stage5}[stage]
    total = sum(check(f, depth) for f in fens)
    print("TOTAL mismatches", total)
