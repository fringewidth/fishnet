# Verification

The claim is that the layers compute the search exactly, not approximately. Three layers of
checks back it.

## The specification

`machine/ref.py` is the search written as a token loop: the exact stream of tokens the layers
must emit. `machine/ref2.py` is the same search with the state laid out the way the layers can
hold it.

- `tests/test_ref.py N`: move generation equals python-chess's on N random positions
  (underpromotions excluded: pawns always promote to queens), and the token machine equals a
  plain recursive alpha-beta (same move, score and node count).
- `tests/test_ref2.py N DEPTH`: `ref2` equals `ref` on N positions.

## The circuit against the specification

These use Qwenputer's IR evaluator, which computes every value in float64 and checks that it
survives fp32 rounding unchanged.

- `tests/test_circuit.py DEPTH STAGE`: every value of every stage against the spec, token by
  token.
- `tests/test_bridge.py DEPTH`: the whole exchange from an all-zero state: Qwen's text, the FEN
  parse, the 64 PUTs, the search and the printed move. 0 mismatches over 13,596 positions,
  covering both colours, partial castling rights and en passant.
- `tests/broad_test.py`: mates, stalemate, promotion and random middlegames.
- `tests/check_all.py DEPTH` and `tests/spans.py`: a check that no fp32 sum can lose precision.
  A sum of fp32 terms is exact in any order only if the sum of their absolute values stays below
  2^24, and the float64 evaluator cannot see a violation on its own.

## The executors against the circuit

- `tests/test_sparse.py DEPTH`: the Mac executor equals the IR on every value at every position.
- `tests/test_exec.py DEPTH`: Qwenputer's batched torch executor (used by `circuit/play.py` on a
  GPU) against the IR.
- `tests/selftest.py`: the layers run alone in a closed loop, feeding back each token they
  predict, and must regenerate the reference stream token for token. `setup.sh` runs this.

## In games

Every move fishnet made in a recorded game was checked by `harness/players.py`. The board the
model loaded (its PUT tokens) must equal the game's board, and its move must equal the reference
search's. The web app does the same check when "check each move against the reference engine"
is ticked.

## Running it

```sh
./tests/run.sh          # the quick suite, about 10 minutes
./tests/run.sh --full   # adds the depth-3 stage checks
```
