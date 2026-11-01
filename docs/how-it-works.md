# How it works

fishnet is Qwen3.5-4B with 23 extra layers in front of its own. The new layers hold an
alpha-beta search (PeSTO evaluation, captures-only quiescence search, depth 4). They are built
from Gated DeltaNet heads and SwiGLU MLPs in the circuit IR of
[Qwenputer](https://github.com/divagr18/qwenputer).

## A move, end to end

1. Qwen is asked for a move and writes a tool call holding the position's FEN:
   `<tool_call>\n<function=chess_engine>\n<parameter=fen>\n...\n</parameter>\n</function>\n</tool_call>`.
1. While Qwen types, the new layers parse the FEN out of its tokens, character by character:
   ranks, files, side to move, castling rights, en passant.
1. From `</tool_call>` onward the new layers own the output. They write 64 PUT tokens (the
   board, one square each), a START token, then the search: every move tried (MOVE), every
   score passed back up (RET), and a DONE token. Each of these tokens is one forward pass.
1. The layers print the chosen move back as Qwen tokens:
   `<tool_response>\ne2e4\n</tool_response>\n`.
1. Qwen reads the response and closes its turn.

No code runs outside the forward pass. The search is about 2 tokens per position searched, so
a typical depth-4 move is 50,000 to 300,000 tokens.

## The three pieces

- Memory. A Gated DeltaNet head keeps a matrix state that is updated by a rank-one write each
  token. A one-hot key with beta = 1 overwrites one row, and a one-hot query reads it back. That
  makes the head addressable RAM. The search keeps one board per ply (64 squares in 4 bit planes),
  plus move cursors, bounds and scores, as rows in these heads.
- Logic. A SwiGLU unit computes `c * a * relu(b)`. With large gains this gives exact
  comparisons, selects and integer adds. Move generation, make-move, check detection, PeSTO
  evaluation and the alpha-beta bookkeeping are all built from these units.
- The clock. Each generated token is one step of the search machine. A layer can only write state
  that it computes in the same pass. Anything that depends on its own previous value therefore
  travels through the token: the next token is the machine's next state.

[design.md](design.md) has the details: the state tables, the order of work inside one pass,
the FEN parser, and the fp32 exactness rules.

## What runs where

`circuit/merge.py` describes the merged checkpoint: the chess layers in fp32, then Qwen's layers,
widened to a 5,888-wide residual stream. Its fp32 weights would be about 66 GB, so on a Mac
`circuit/play_mac.py` computes the same function without materializing the zero blocks:

- Qwen's layers run as shipped (bf16, on the GPU through MPS).
- The chess layers run on the CPU through `circuit/sparse_exec.py`, which skips zero inputs and
  is exact against the IR.
- Chess tokens skip Qwen's layers (transparent decoding, as in Qwenputer's `rv/chat.py`). In the
  merged model Qwen is muted on those positions, so they contribute nothing to its output.

Not done yet: writing the merged checkpoint and running its chess layers through the stock
Hugging Face `Qwen3_5` modules, layer by layer, on recorded game streams.

## Layout

- `app/` the localhost web app (`server.py`, `index.html`), started by `play.sh`.
- `circuit/` the compiled engine:
  - `build.py` the circuit
  - `text.py` and `vocab.py` Qwen's tokens and the 161,331 chess tokens
  - `play_mac.py` and `sparse_exec.py` the Mac runtime
  - `play.py` and `merge.py` the GPU runtime and the merged checkpoint
- `machine/` the specification: `ref.py` (the token-level search), `ref2.py` (the same search
  in the state layout the layers hold), `pesto.py`.
- `tests/` the verification suite ([verification.md](verification.md)).
- `harness/` games against Claude Opus 5.5 and the Stockfish calibration ([matches.md](matches.md)).
- `video/` replay videos from game records.
- `results/` game records.
