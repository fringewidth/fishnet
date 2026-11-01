"""Stage 6: a whole exchange through the exact IR evaluator from an all-zero state.

Qwen's text (a prompt, then a tool call holding a FEN) -> the circuit owns </tool_call> onward:
64 PUTs, START, the search (spec v2's tokens), DONE, the echo of the best move as text -> Qwen
again, and a second tool call with another position. Every circuit-owned prediction must be the
stream's next token; everywhere else the circuit must leave the output code at 0 and tau at
Qwen's value.

    python test_bridge.py [DEPTH]
"""
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/machine")
import build as B  # noqa: E402
import ref  # noqa: E402
import ref2  # noqa: E402
import vocab as V  # noqa: E402
from chipformer.q35 import ir  # noqa: E402
from text import Text  # noqa: E402

CALL = "<tool_call>\n<function=chess_engine>\n<parameter=fen>\n{fen}\n</parameter>\n</function>\n</tool_call>"


def uci(m) -> str:
    _, f, t, pk, xk, white = m
    sq = lambda s: "abcdefgh"[s % 8] + str(s // 8 + 1)  # noqa: E731
    return sq(f) + sq(t) + ("q" if pk == ref.P and t // 8 in (0, 7) else "")


def exchange(T: Text, fen: str, depth: int, before: str):
    """(tokens, owned flags) for: Qwen's text `before` + the tool call, then the circuit's run."""
    toks, own = [], []
    for q in T.encode(before + CALL.format(fen=fen)):
        toks.append(V.TEXT0 + q)
        own.append(q == T.TCC)
    pos, white = ref.from_fen(fen)
    m = ref2.Machine2(pos, white, depth)
    m.run()
    board = ref2.abs_board(pos, white)
    for s in range(64):
        toks.append(V.encode(("PUT", s, board.get(s, 0))))
        own.append(True)
    toks += [V.encode(t) for t in m.tokens]
    own += [True] * len(m.tokens)
    mv = uci(m.bm)
    echo = ["<tool_response>", "\n", *mv, "\n", "</tool_response>", "\n"]
    for i, ch in enumerate(echo):
        toks.append(V.TEXT0 + T.print_ids[{v: k for k, v in B_PRINT.items()}[ch]])
        own.append(i < len(echo) - 1)
    return toks, own, m, mv


def main():
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    T = Text()
    t0 = time.time()
    c, (s1, s2, s3, s4, s5, br, s6) = B.build_all(T, depth)
    print(f"built {c.stats()} in {time.time() - t0:.0f}s", flush=True)
    fens = ["r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "8/P7/8/8/8/k7/8/7K w - - 0 1",
            "r3k2r/ppp2ppp/2nq1n2/3pp3/1b1PP1b1/2N2N2/PPPQBPPP/R3K2R b Qk d3 0 8",
            "8/8/8/8/2pP4/8/1k6/4K3 b - d3 0 1"]
    prompt = ("<|im_start|>system\nYou are a chess player with a chess engine tool.<|im_end|>\n"
              "<|im_start|>user\nPosition (FEN): {fen}\nYour move?<|im_end|>\n<|im_start|>assistant\n")
    toks, own, runs = [], [], []
    for i, fen in enumerate(fens):
        pre = prompt.format(fen=fen) if i == 0 else " That is my move.\n\nNext position:\n"
        t, o, m, mv = exchange(T, fen, depth, pre)
        toks += t
        own += o
        runs.append((fen, mv, len(m.tokens)))
    print(f"stream {len(toks)} tokens; runs {runs}", flush=True)
    t0 = time.time()
    mat = ir.evaluate(c, toks, check=True)
    print(f"evaluated in {time.time() - t0:.0f}s", flush=True)
    code = np.stack([ir.ev(e, mat) for e in c.code], 1)
    owner = ir.ev(s6["owner"], mat)
    tau = ir.ev(c.tau, mat)
    bad = 0
    for i in range(len(toks) - 1):
        if own[i]:
            if owner[i] != 1 or not np.all(np.abs(code[i]) == B.KAPPA) or tau[i] != 0:
                got = None
            else:
                got = B.decode_code([int(v > 0) for v in code[i]], T)
            if got != toks[i + 1]:
                bad += 1
                if bad <= 10:
                    want = V.decode(toks[i + 1])
                    print(f"  pos {i} after {V.decode(toks[i])}: want {want}, got "
                          f"{V.decode(got) if got not in (None, -1) else got} (owner {owner[i]})")
        else:
            want_tau = B.TAU_TEXT if toks[i] >= V.TEXT0 else 0
            if owner[i] != 0 or np.any(code[i] != 0) or tau[i] != want_tau:
                bad += 1
                if bad <= 10:
                    print(f"  pos {i} {V.decode(toks[i])} (Qwen's): owner {owner[i]}, code nonzero "
                          f"{int(np.sum(code[i] != 0))}, tau {tau[i]}")
    print(f"BRIDGE mismatches {bad} over {len(toks) - 1} positions ({sum(own)} circuit-owned)")


B_PRINT = __import__("text").PRINT
if __name__ == "__main__":
    main()
