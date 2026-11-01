"""The LLM opponent: one fresh `claude -p` session per move, board tools only."""
import json
import os
import subprocess
import tempfile
import time

import chess

HERE = os.path.dirname(os.path.abspath(__file__))

SYSTEM = """You are playing a game of chess as {color}. Play the strongest move you can.
You have three tools: get_board (the full position), legal_moves (every legal move), and
submit_move (play your move). You have no other tools and no chess engine. Think as long as you
like, then call submit_move exactly once with your chosen move."""


class ClaudePlayer:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "high", log_dir: str | None = None):
        self.model, self.effort = model, effort
        self.name = f"{model}({effort})"
        self.work = tempfile.mkdtemp(prefix="llm-chess-")
        self.log_dir = log_dir
        self.last = ""
        self.cfg = os.path.join(self.work, "mcp.json")

    def move(self, board: chess.Board) -> chess.Move:
        state = os.path.join(self.work, "state.json")
        out = os.path.join(self.work, "move.txt")
        json.dump({"start_fen": board.root().fen(), "moves": [m.uci() for m in board.move_stack]}, open(state, "w"))
        if os.path.exists(out):
            os.remove(out)
        json.dump({"mcpServers": {"chess": {
            "command": "uv", "args": ["run", "-q", "--no-project", "--with", "mcp<2", "--with", "chess", "python",
                                      os.path.join(HERE, "board_mcp.py")],
            "env": {"CHESS_STATE": state, "CHESS_OUT": out}}}}, open(self.cfg, "w"))
        color = "White" if board.turn else "Black"
        from board_mcp import describe
        prompt = f"It is your move ({color}).\n\n{describe(board)}\n\nChoose your move and submit it."
        cmd = ["claude", "-p", prompt, "--model", self.model, "--effort", self.effort,
               "--system-prompt", SYSTEM.format(color=color), "--tools", "",
               "--strict-mcp-config", "--mcp-config", self.cfg,
               "--allowedTools", "mcp__chess__get_board,mcp__chess__legal_moves,mcp__chess__submit_move",
               "--no-session-persistence", "--output-format", "json"]
        for attempt in range(3):
            t0 = time.time()
            r = subprocess.run(cmd, cwd=self.work, capture_output=True, text=True, timeout=3600)
            dt = time.time() - t0
            if self.log_dir:
                n = len(board.move_stack)
                open(os.path.join(self.log_dir, f"ply{n:03d}-try{attempt}.json"), "w").write(r.stdout + r.stderr)
            if os.path.exists(out):
                self.last = f"{dt:.0f}s"
                return chess.Move.from_uci(open(out).read().strip())
        raise RuntimeError(f"no move submitted after 3 attempts: {r.stdout[-500:]} {r.stderr[-500:]}")

    def close(self):
        pass
