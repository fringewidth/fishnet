"""Players for match.py and calibrate.py. Each has .name, .last (a note on its last move) and
.move(board) -> chess.Move."""
import gzip
import json
import os
import shutil
import sys
import urllib.request

import chess
import chess.engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Reference:
    """The token-level specification of the compiled search (machine/ref.py) at a fixed depth.
    The weights must reproduce its move exactly, so it plays the same chess as fishnet, fast."""

    def __init__(self, depth: int):
        sys.path.insert(0, os.path.join(ROOT, "machine"))
        import ref
        self.ref = ref
        self.depth = depth
        self.name = f"reference-search@depth{depth}"
        self.last = ""

    def move(self, board: chess.Board) -> chess.Move:
        pos, white = self.ref.from_fen(board.fen())
        m = self.ref.Machine(pos, self.depth)
        done = m.run()
        f, t = self.ref.real_square(done[1], white), self.ref.real_square(done[2], white)
        promo = chess.QUEEN if board.piece_type_at(f) == chess.PAWN and chess.square_rank(t) in (0, 7) else None
        self.last = f"score {done[3]} nodes {m.nodes} tokens {len(m.tokens)}"
        return chess.Move(f, t, promotion=promo)

    def close(self):
        pass


class Fishnet:
    """The model itself, served by `circuit/play_mac.py --serve PORT` (or circuit/play.py on a
    GPU). Every move is checked: the board the model loaded (its PUT tokens) must equal the
    game's, and its move must equal the reference search's at the same depth."""

    def __init__(self, url: str = "http://127.0.0.1:8765", log_dir: str | None = None):
        self.url, self.log_dir = url, log_dir
        self.name = "fishnet (Qwen3.5-4B, in weights)"
        self.last = ""
        self.detail = None

    def move(self, board: chess.Board) -> chess.Move:
        req = urllib.request.Request(self.url, data=json.dumps({"fen": board.fen()}).encode(),
                                     headers={"Content-Type": "application/json"})
        res = json.loads(urllib.request.urlopen(req, timeout=3600).read())
        if "error" in res:
            raise RuntimeError(res["error"])
        if self.log_dir:  # the whole token stream of the move, for the replay video
            path = os.path.join(self.log_dir, f"ply{len(board.move_stack):03d}-fishnet.json.gz")
            with gzip.open(path, "wt") as fh:
                json.dump(res, fh)
        tokens = res.pop("tokens")
        res["n_tokens"] = len(tokens)
        self.detail = res
        want = {s: p.piece_type + (0 if p.color else 6) for s, p in board.piece_map().items()}
        got = {s: code for _, s, code in res["puts"] if code}
        assert got == want, ("the board the model loaded differs from the game's", board.fen())
        depth = res["start"][1]
        ref_move = Reference(depth).move(board).uci()
        assert res["move"] == ref_move, ("the model's move differs from the reference", res["move"], ref_move,
                                         board.fen())
        self.last = (f"{res['move']} depth {depth} nodes {res['nodes']} "
                     f"tokens {res['chip_tokens']}+{res['text_tokens']} {res['seconds']}s")
        return chess.Move.from_uci(res["move"])

    def close(self):
        pass


class Stockfish:
    """Stockfish with UCI_LimitStrength, from $STOCKFISH or the PATH (brew install stockfish)."""

    def __init__(self, elo: int, movetime: float = 1.0):
        path = os.environ.get("STOCKFISH") or shutil.which("stockfish")
        if not path:
            raise FileNotFoundError("no stockfish binary: brew install stockfish, or set STOCKFISH")
        self.e = chess.engine.SimpleEngine.popen_uci(path)
        self.e.configure({"UCI_LimitStrength": True, "UCI_Elo": elo, "Threads": 1, "Hash": 16})
        self.name = f"stockfish@{elo}"
        self.limit = chess.engine.Limit(time=movetime)
        self.last = ""

    def move(self, board: chess.Board) -> chess.Move:
        return self.e.play(board, self.limit).move

    def close(self):
        self.e.quit()
