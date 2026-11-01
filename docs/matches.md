# Matches

## Against Claude Opus 5.5

Opus plays through `claude -p` (no API key needed; it uses your Claude Code login). Each move is
a fresh session with high effort, the position in the prompt, and three tools from
`harness/board_mcp.py`: `get_board`, `legal_moves`, `submit_move`. It has no engine and no
code execution.

```sh
uv sync --group harness
uv run python circuit/play_mac.py --serve 8765 &          # the model, kept loaded
uv run python harness/match.py fishnet white my-game       # fishnet plays White
```

`harness/match.py d4 black my-game` plays the reference search instead of the model. It
makes the same moves (the model is checked against it) in a fraction of the time.

Each game writes, under `harness/`:

- `records/TAG.jsonl`: one line per ply, with the position, the move, the timing, Opus's
  event stream and the model's statistics.
- `games/TAG.pgn`
- `logs/TAG/`: Opus's raw event streams and the model's full token stream for each move.

`--resume` continues a stopped game from its record.

## Calibration against Stockfish

```sh
brew install stockfish
uv run python harness/calibrate.py 4 1600 10 1.0   # depth 4 vs Stockfish UCI_Elo 1600, 10 games, 1 s/move
```

This uses the reference search, which plays exactly the model's moves. Colours alternate over
ten balanced openings.

## Replay video

```sh
uv sync --group video && uv run playwright install chromium
uv run python video/build_timeline.py my-game --pace tweet > video/my-game.json
uv run python video/render.py video/my-game.json video/my-game.mp4
```

`--pace full` gives every move the same screen time. `tweet` compresses the middlegame.
