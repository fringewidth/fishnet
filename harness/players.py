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
