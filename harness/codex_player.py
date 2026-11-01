"""The OpenAI opponent: one fresh `codex exec` session per move with board tools only.

Built-in shell, browser, computer-use, image, multi-agent, plugins, apps and web search are
disabled, and the user's node_repl MCP server is switched off, so the only tools are the chess
MCP server's get_board / legal_moves / submit_move. Each run's JSON event log is kept, and
tool_audit() lists every tool the model called."""
import json
import os
import subprocess
import tempfile
import time

import chess

from claude_player import SYSTEM

HERE = os.path.dirname(os.path.abspath(__file__))
DISABLE = ["shell_tool", "browser_use", "browser_use_external", "computer_use", "image_generation",
           "multi_agent", "plugins", "apps", "in_app_browser", "goals", "sleep_tool", "tool_suggest",
           "skill_search"]


def toml_str(s: str) -> str:
    return json.dumps(s)


class CodexPlayer:
    def __init__(self, model: str = "gpt-6-astra", effort: str = "high", log_dir: str | None = None):
        self.model, self.effort = model, effort
        self.name = f"{model}({effort})"
        self.work = tempfile.mkdtemp(prefix="llm-chess-codex-")
        self.log_dir = log_dir
        self.last = ""
        self.violations = []

    def move(self, board: chess.Board) -> chess.Move:
        from board_mcp import describe
        state = os.path.join(self.work, "state.json")
        out = os.path.join(self.work, "move.txt")
        json.dump({"start_fen": board.root().fen(), "moves": [m.uci() for m in board.move_stack]}, open(state, "w"))
        if os.path.exists(out):
            os.remove(out)
        color = "White" if board.turn else "Black"
        prompt = (SYSTEM.format(color=color) + f"\n\nIt is your move ({color}).\n\n{describe(board)}\n\n"
                  "Choose your move and submit it with the submit_move tool.")
        args = ["run", "-q", "--no-project", "--with", "mcp<2", "--with", "chess", "python",
                os.path.join(HERE, "board_mcp.py")]
        cmd = ["codex", "exec", "-m", self.model, "-c", f"model_reasoning_effort={toml_str(self.effort)}",
               "-c", 'web_search="disabled"', "-c", "mcp_servers.node_repl.enabled=false",
               "-c", 'mcp_servers.chess.command="uv"',
               "-c", "mcp_servers.chess.args=[" + ",".join(toml_str(a) for a in args) + "]",
               "-c", f"mcp_servers.chess.env={{CHESS_STATE={toml_str(state)},CHESS_OUT={toml_str(out)}}}",
               "-c", 'mcp_servers.chess.default_tools_approval_mode="approve"',
               "-s", "read-only", "--skip-git-repo-check", "--ephemeral", "--json"]
        for f in DISABLE:
            cmd += ["--disable", f]
        cmd.append(prompt)
        for attempt in range(3):
            t0 = time.time()
            r = subprocess.run(cmd, cwd=self.work, capture_output=True, text=True, timeout=3600, stdin=subprocess.DEVNULL)
            dt = time.time() - t0
            if self.log_dir:
                n = len(board.move_stack)
                open(os.path.join(self.log_dir, f"ply{n:03d}-try{attempt}.jsonl"), "w").write(r.stdout)
                open(os.path.join(self.log_dir, f"ply{n:03d}-try{attempt}.err"), "w").write(r.stderr)
            bad = violations(r.stdout)
            if bad:
                self.violations.append((len(board.move_stack), attempt, bad))
                if os.path.exists(out):
                    os.remove(out)  # a move found with any non-board tool is discarded
                continue
            if os.path.exists(out):
                self.last = f"{dt:.0f}s"
                return chess.Move.from_uci(open(out).read().strip())
        raise RuntimeError(f"no move submitted after 3 attempts: {r.stdout[-800:]} {r.stderr[-800:]}")

    def close(self):
        pass


ALLOWED = {("chess", "get_board"), ("chess", "legal_moves"), ("chess", "submit_move")}


def violations(jsonl_text: str) -> list[str]:
    """Every item in a codex --json event log that is not a message, reasoning, or one of the
    three chess board tools (shell commands, code execution, web search, other MCP tools)."""
    bad = []
    for line in jsonl_text.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        item = ev.get("item") or {}
        t = item.get("type")
        if t is None or t in ("agent_message", "reasoning"):
            continue
        if t == "mcp_tool_call" and (item.get("server"), item.get("tool")) in ALLOWED:
            continue
        if t == "error":
            continue
        bad.append(f"{t}:{item.get('server', '')}:{item.get('tool') or item.get('command') or ''}")
    return bad
