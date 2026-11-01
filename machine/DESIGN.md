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

## Status

Stages 1-5 exact against the spec (22 layers before depth optimisation, ~40k MLP units, 24
heads). Next: the bridge to Qwen (FEN read from text via width counters into summed root rows;
START; printing bestmove), the output code, emit / merge into Qwen3.5-4B, a GPU run.
