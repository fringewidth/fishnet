# fishnet: a chess search machine compiled into transformer layers

The spec is `ref.py` (token-level reference, checked against a recursive alpha-beta and against
python-chess move generation). The circuit must emit exactly `ref.Machine`'s token stream.

## Arithmetic

fp32, not bf16. Every value the circuit materializes is an integer below 2^24, so fp32 holds it
exactly, and the repo's exact-relu rule carries over unchanged: silu(128 x) == 128 relu(x) for
integer x (x >= 1 gives >= 18, x <= -1 gives <= -104). That makes scores, alpha-beta, max/min
and the PeSTO sums plain linear forms plus single relu units, where bf16 would need nibble
logic everywhere. State rows still hold 0/1 bits next to the ballast (the per-head RMSNorm only
absorbs small data next to it), so integers are written as bits and read back linearly.

## State (Gated DeltaNet tables, keyed by ply unless noted)

| table  | rows        | contents                                                   | bits |
|--------|-------------|------------------------------------------------------------|------|
| board  | ply 0..31   | 64 squares x 12 piece one-hot, castling 4, ep square 6 + valid | 779  |
| ctrl   | ply         | alpha, beta, best (16-bit, offset), depth (5), legal (1)   | ~54  |
| cursor | ply         | last move tried (from 6, to 6, valid 1)                    | 13   |
| ply    | one row     | current ply (5 bits)                                       | 5    |
| rootbm | one row     | root's best move so far                                    | 12   |

Each value head row holds 126 data bits, so the board row spans 7 value heads (4 key heads that
all key on the same ply).

## One pass (one token)

The input token is START(depth), MOVE(f, t), RET(s), RET_ILLEGAL or DONE.

1. Read the ply, then the rows of the focus node: the parent (MOVE: ply; RET: ply - 1).
2. MOVE: build the child board in the residual (move the piece, captures, en passant, castling
   rook, queen promotion, rights, new ep; mirror sq ^ 56 and swap colours, a pure rewiring) and
   its ctrl (alpha = -beta_p, beta = -alpha_p, depth - 1). RET: v = -s; best = max(best, v);
   alpha = max(alpha, v); legal += 1; cutoff = v >= beta. Mux into one "node" board + ctrl.
3. Node logic on the node board, all squares in parallel:
   - attack maps (path-clear ANDs over every ray), king capture (illegal), in check, castling;
   - eval (MOVE into a depth <= 0 child): mg, eg and phase are linear in the one-hot board;
     mg * ph by phase one-hot; floor(/24) by thresholds; stand pat, cutoff;
   - next move after the cursor: captures by (victim value desc, victim sq, attacker value
     asc, attacker sq), then quiets by (from, to); two-level selection (victim, then attacker;
     from, then to) with a priority encoder each.
4. Write the rows that changed (child board / ctrl / parent cursor on MOVE; parent ctrl on RET;
   ply; root best move) and emit the next token's output code.

## Tokens per node

Exactly 2 (one MOVE in, one RET out), measured on the reference.
