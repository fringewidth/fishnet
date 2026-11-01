# fishnet

A chess engine compiled into the weights of a language model.

fishnet is Qwen3.5-4B with 23 extra layers in front of its own. The new layers hold an
alpha-beta search (PeSTO evaluation, quiescence search, depth 4), built from Gated DeltaNet heads
and SwiGLU MLPs in the circuit IR of [Qwenputer](https://github.com/divagr18/qwenputer). Qwen
plays chess the way it would call any tool: it writes a tool call with the position's FEN. The
"tool" is the model itself. From `</tool_call>` onward the new layers own the output and write
the search one token per forward pass: the board, then every move tried and every score returned,
then the chosen move as text. Qwen reads the result and answers. No code runs outside the forward
pass.

## Play it on your Mac

You need an Apple Silicon Mac with 16 GB of memory or more,
[uv](https://docs.astral.sh/uv/getting-started/installation/), git, and about 10 GB of disk.

```sh
git clone https://github.com/fringewidth/fishnet && cd fishnet
./setup.sh     # Qwenputer's IR, the Python env, Qwen3.5-4B (9 GB), a self-test
./play.sh      # opens http://127.0.0.1:8000
```

Click a piece, then a square. On its turn the page shows the model at work:

1. Qwen writes the tool call.
1. The chess layers write the search. Every MOVE, RET and PLY token in the panel is one forward
   pass, with a running count of tokens and positions.
1. The move comes back in a `<tool_response>`, and Qwen closes its turn.

A move takes 10 seconds to 3 minutes on an M4 Pro, about 45 seconds typically (each search token takes about 0.4 ms).
Tick "check each move against the reference engine" to have the server rerun the same search in
plain Python (`machine/ref.py`) after every move and confirm the two agree.

`./play.sh --depth 3` plays faster and weaker. `--port` and `--no-open` do what they say.

## Results

| game | result |
|---|---|
| reference search, depth 4, vs Claude Opus 5.5 (high effort), both colours | 2-0 |
| fishnet (in the weights) vs Claude Opus 5.5 (high effort), fishnet White | in progress |
| reference search, depth 4, vs Stockfish 16 at `UCI_Elo` 1320 / 1600 / 1900, 1 s/move | in progress |

Opus played through an MCP server that shows the board, the legal moves, and the move history
(`harness/board_mcp.py`), and nothing else: no engine, no code execution. In every game fishnet
played, the harness checked two things on each move. The board the model wrote in its PUT tokens
must equal the real board. The move must equal the reference search's move.

## How it works

Three pieces, all in `circuit/build.py`:

- Memory. A Gated DeltaNet head is a matrix state updated by a rank-one write each token. A
  one-hot key with beta = 1 overwrites one row, and a one-hot query reads it back. That makes it
  addressable RAM. The search keeps one board per ply (64 squares in 4 bit planes), plus move
  cursors, bounds and scores, as rows in these heads.
- Logic. SwiGLU units compute `c * a * relu(b)`. With large gains that becomes exact
  comparisons, selects and integer adds. Move generation, make-move, check detection, PeSTO
  evaluation and the alpha-beta bookkeeping are all built from them. Wide sums are split into
  digit stages so that every partial sum stays below 2^24 and fp32 arithmetic stays exact.
- The clock. Each generated token is one step of the search machine. A layer can only write state
  that it computes in the same pass, so anything that depends on its own previous value travels
  through the token: the next token is the machine's next state.

The text interface is in the weights as well. The layers parse the FEN out of Qwen's tool call
character by character: ranks, files, side to move, castling rights, en passant. They print the
chosen move back as Qwen tokens.

`machine/DESIGN.md` has the full design. `machine/ref.py` is the token-level specification the
circuit is checked against.

## Verification

- `circuit/selftest.py` (run by setup.sh). The layers generate a whole search on their own,
  feeding back each token they predict, and the stream must be identical to `machine/ref.py`'s.
- `circuit/test_bridge.py`. The full exchange (Qwen's text, the FEN parse, the search, the
  printed move) through Qwenputer's exact IR evaluator. Zero mismatches over 13,596 positions,
  covering both colours, partial castling rights and en passant.
- `circuit/check_all.py` and `circuit/spans.py`. Every intermediate value in every layer, IR
  versus fp32 execution, and a check that no fp32 sum can lose precision.
- `circuit/test_sparse.py`. The Mac executor against the IR on every value at every position.

## What runs where

`circuit/merge.py` writes the merged checkpoint: the chess layers in fp32, then Qwen's layers,
widened to a 5,888-wide residual stream. That is the artifact the claim is about. Its fp32
weights are about 66 GB, so on a Mac `circuit/play_mac.py` computes the same function without
materializing the zero blocks:

- Qwen's layers run as shipped (bf16, MPS).
- The chess layers run on the CPU through `circuit/sparse_exec.py`, which skips zero inputs and
  is exact against the IR.
- Chess tokens skip Qwen's layers (transparent decoding, as in Qwenputer's `rv/chat.py`). In the
  merged model Qwen is muted on those positions, so they contribute nothing to its output.

Pending: running the emitted chess layers through the stock Hugging Face `Qwen3_5` modules,
layer by layer, on recorded game streams.

## Layout

- `app/` the localhost web app (`server.py`, `index.html`).
- `circuit/` the compiled engine: `build.py` (the circuit), `text.py` (Qwen's tokens),
  `vocab.py` (161,331 chess tokens), `play_mac.py` and `sparse_exec.py` (the Mac runtime),
  `play.py` and `merge.py` (GPU runtime and merged checkpoint), tests.
- `machine/` the specification: `ref.py`, `ref2.py` (the state layout the layers hold),
  `pesto.py`, `DESIGN.md`.
- `harness/` matches against Claude Opus 5.5 (`claude -p`) and GPT-6 Astra (`codex exec`), and
  the Stockfish calibration (`calibrate.py`). Install the extras with `uv sync --group harness`.
- `video/` replay video from a game's records (`uv sync --group video`).
- `legacy/rv32/` the first attempt: a C engine compiled to RISC-V firmware for the published
  Qwenputer checkpoint (perft-verified). It was too slow at a few nodes per minute, which is why
  the dedicated machine exists.

## Credits

- [Qwenputer](https://github.com/divagr18/qwenputer) (MIT) by divagr18: the circuit IR, the
  Gated DeltaNet RAM construction, and the merge approach. setup.sh fetches it pinned to
  `bec1a0c`.
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache-2.0).
- PeSTO piece-square tables by Ronald Friederich.
- [python-chess](https://github.com/niklasf/python-chess) for move legality in the harness and
  the web app.
- Stockfish is used only as a calibration opponent. It is not part of fishnet.
