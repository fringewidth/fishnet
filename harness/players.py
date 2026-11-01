"""Players for match.py. Each has .name and .move(board) -> chess.Move."""
import subprocess

import chess
import chess.engine

ROOT = __file__.rsplit("/", 2)[0]


class Qwenputer:
    """The firmware's engine, native build (byte-identical responses to the RV32 build)."""

    def __init__(self, nodes: int):
        self.nodes = nodes
        self.name = f"qwenputer-native@{nodes}"
        self.p = subprocess.Popen([f"{ROOT}/chess_native", "search", str(nodes)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        self.last = ""

    def move(self, board: chess.Board) -> chess.Move:
        self.p.stdin.write(board.fen() + "\n")
        # the response is "bestmove X" + "depth ..." lines, or a single error / no-moves line
        lines = [self.p.stdout.readline().rstrip("\n")]
        if lines[0].startswith("bestmove "):
            lines.append(self.p.stdout.readline().rstrip("\n"))
        self.last = " | ".join(lines)
        assert lines and lines[0].startswith("bestmove "), self.last
        return chess.Move.from_uci(lines[0].split()[1])

    def close(self):
        self.p.kill()


class Stockfish:
    def __init__(self, elo: int, movetime: float = 0.2):
        self.name = f"stockfish16@{elo}"
        self.e = chess.engine.SimpleEngine.popen_uci(["docker", "run", "--rm", "-i", "rvchess", "/usr/games/stockfish"])
        self.e.configure({"UCI_LimitStrength": True, "UCI_Elo": elo, "Threads": 1, "Hash": 16})
        self.limit = chess.engine.Limit(time=movetime)
        self.last = ""

    def move(self, board: chess.Board) -> chess.Move:
        return self.e.play(board, self.limit).move

    def close(self):
        self.e.quit()


class SearchMachine:
    """The token-level reference of the compiled search machine (machine/ref.py), fixed depth.
    The circuit must emit exactly this token stream, so this plays the in-weights moves."""

    def __init__(self, depth: int):
        import sys
        sys.path.insert(0, f"{ROOT}/machine")
        import ref
        self.ref = ref
        self.depth = depth
        self.name = f"qwenputer-machine@d{depth}"
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
    """The merged checkpoint (Qwen3.5-4B with the search machine in its weights), served by
    circuit/play.py on a GPU box and reached through an SSH tunnel. Every move also checks the
    weights' run against the token-level spec: the move must equal SearchMachine's at the
    checkpoint's depth, and the board Qwen loaded (its PUT tokens) must equal the real one."""

    def __init__(self, url: str = "http://127.0.0.1:8765", verify: int = 0):
        self.url, self.verify = url, verify
        self.name = "fishnet (Qwen3.5-4B, in weights)"
        self.last = ""
        self.log = []

    def move(self, board: chess.Board) -> chess.Move:
        import json
        import urllib.request
        req = urllib.request.Request(self.url, data=json.dumps({"fen": board.fen(), "verify": self.verify}).encode(),
                                     headers={"Content-Type": "application/json"})
        res = json.loads(urllib.request.urlopen(req, timeout=3600).read())
        if "error" in res:
            raise RuntimeError(res["error"])
        self.log.append(res)
        want = {s: (p.piece_type + (0 if p.color else 6)) for s, p in board.piece_map().items()}
        got = {s: code for _, s, code in res["puts"] if code}
        assert got == want, ("the board Qwen loaded differs from the game's", board.fen())
        depth = res["start"][1]
        spec = SearchMachine(depth).move(board).uci()
        assert res["move"] == spec, ("the weights' move differs from the spec's", res["move"], spec, board.fen())
        self.last = (f"{res['move']} depth {res['start'][1] if res['start'] else '?'} nodes {res['nodes']} "
                     f"tokens {res['chip_tokens']}+{res['text_tokens']} {res['seconds']}s")
        return chess.Move.from_uci(res["move"])

    def close(self):
        pass
