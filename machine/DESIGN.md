# fishnet: a chess search machine compiled into transformer layers

The spec is `ref.py` (token-level search, checked against recursive alpha-beta and python-chess
move generation) and `ref2.py` (the same search with the state layout transformer layers can
hold; same move, score and node count as `ref.py`). The circuit (`circuit/build.py`) predicts
exactly `ref2.Machine2`'s token stream: `circuit/test_circuit.py 2 5` checks every next-token
prediction with the repo's exact IR evaluator in fp32.

## The one rule that shapes the design

A Gated DeltaNet head's write may only use values computed below its own layer in the same
pass. So no table can be rewritten from its own contents in one pass; such state is carried by
tokens (the CPU in the Qwenputer repo does the same for its registers and PC).

## Arithmetic

fp32. Every value is an integer below 2^24, and silu(128 x) == 128 relu(x) holds for integer x
in fp32 as in bf16, so threshold units are exact: `[z >= theta] = relu(z - theta + 1) -
relu(z - theta)`, with 0/1 literals folded in at weight 2^20. State rows hold 0/1 bits next to
a ballast; summed reads (every row ballasted, a power-of-2 row count) return column sums.

## Tokens (vocab 160,499)

START(depth, root colour, castling, ep file); MOVE / MOVE_IMP(from, to, moved kind, captured
kind [6 = en passant], mover colour) over geometrically possible moves; RET(score) and
RET_ILLEGAL; PLY / PLY_IMP(ply) anchors; DONE. About 2.04 tokens per searched node.

## State tables

| table | layout | written by |
|---|---|---|
| anchor, counters | ply = anchor + MOVEs - RETs since the anchor (decay 1/2 counters) | anchor tokens; every MOVE / RET |
| diff | per ply: a move's add / remove code bits per square + rights lost; board = sum of rows | MOVE (from the token alone); cleared by RET |
| ep, cursor | per ply, single-row reads | MOVE (from the token) |
| root | depth, root colour, castling | START |
| ext | per ply check-extension flag, summed: depth = D - ply + sum | fresh passes (high layer); cleared by RET |
| stand | per-ply 16-bit stand pat in its own column block (one read gives every ancestor's) | fresh quiescence passes (high layer) |
| best (x2, by parity) | per-ply 16-bit best child score, column blocks | reset when a node is created; an improvement is written by the token after the RET (MOVE_IMP / PLY_IMP), from the RET's payload through a lag head |

Scores are stored as score + 32768; the all-zero row (-32768) is -INF.

## One pass

1. Ply (counters), then the tables. 2. The board: diff sum -> absolute one-hot -> the side to
move's view (Black mirrored). 3. Attack maps (path-clear AND units over every ray), legality
(king capture), in check, castling, en passant. 4. PeSTO eval (linear in the one-hot board;
mg * phase via a phase one-hot; floor(/24) in two threshold stages), depth, stand pat, the
alpha-beta window (masked max trees over ancestors). 5. Move selection after the cursor: (a)
same victim, next attacker; (b) next victim, first attacker; (c) same from-square, next quiet
target; (d) next from-square with a quiet move, first target; 384-slot priority encoders over
(victim value desc, square) and (attacker value asc, square). 6. Cases (illegal, ply cap, stand
pat cut, beta cut, move, no move / mate / stalemate, root DONE, anchors) -> the next token.

## The bridge (stage 6)

Qwen writes a tool call holding the FEN; the circuit reads it from Qwen's own tokens. A line
starts after a newline; its field is the number of spaces so far, and it turns dirty at a token
a FEN cannot hold. In field 0 of a clean line inside <tool_call>, a square cursor (four counters,
one per bit of each token's width) gives each token's first square, and the token writes its
squares into 8 per-file tables (a token covers at most 8 consecutive squares, so at most one per
file). Fields 1-3 set side, castling and en passant. At </tool_call> the circuit owns the
stream: 64 PUT(square, piece) tokens load the root board (one row per square, read at the bottom
layer like the diff stack), START(depth, side, castling, ep) starts the search, a BM row records
the root move at every root improvement, and after DONE the circuit prints
"<tool_response>\ne2e4\n</tool_response>\n" as Qwen tokens; Qwen speaks again after it.

The model starts from an all-zero state (no snapshot): every summed table keeps its ballast in
one extra row written at a setup token (TCO for the FEN tables, TCC for the root board, PUT(0)
for the search tables), and reads are 0 before it.

## fp32 summation spans

The IR evaluator sums in float64. The hardware sums every projection and MLP output in some order
in fp32, which is exact only if the sum of the terms' absolute values stays below 2^24. A
threshold node `sum_j [x >= theta_j]` over a wide x violates this (the relu pairs cancel only at
the end), so wide divisions and bit extractions are done in digit stages (`ChessCirc.digit`),
each with a small span; the digit stages also give the bits of their digit directly.
`circuit/spans.py` measures every span on an evaluated stream.

## Status

Stages 1-6 exact against the spec: `test_bridge.py` (0 mismatches over 13,596 positions of five
exchanges, from an all-zero state), the repo's fast executor equal to the IR evaluator, spans
below 2^24. 23 layers, ~42k MLP units, 51 heads, 3,201 residual slots: the merged model has
hidden size 5,888. Next: `merge.py` on a GPU box (fp32 chip layers), `play.py` with
`--verify` (HF modules == executor) and `--check-head`, then games.
