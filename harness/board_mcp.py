"""MCP server giving the LLM opponent board inspection and nothing else.

State comes from the JSON file in $CHESS_STATE ({"start_fen", "moves": [uci...]}); the chosen
move is written to $CHESS_OUT. Tools:
  get_board      ASCII diagram, FEN, side to move, check status, castling, en passant, history
  legal_moves    every legal move in SAN and UCI
  submit_move    play a move (SAN or UCI); illegal input returns an error and may be retried
"""
import json
import os

import chess
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("chess")


def load() -> chess.Board:
    st = json.load(open(os.environ["CHESS_STATE"]))
    b = chess.Board(st.get("start_fen", chess.STARTING_FEN))
    for u in st["moves"]:
        b.push_uci(u)
    return b


def describe(b: chess.Board) -> str:
    rows = []
    for r in range(7, -1, -1):
        cells = []
        for f in range(8):
            p = b.piece_at(chess.square(f, r))
            cells.append(p.symbol() if p else ".")
        rows.append(f"{r + 1} | " + " ".join(cells))
    diagram = "\n".join(rows) + "\n    ---------------\n    a b c d e f g h"
    replay = chess.Board(b.root().fen())
    san = []
    for m in b.move_stack:
        san.append(replay.san(m))
        replay.push(m)
    hist = " ".join((f"{i // 2 + 1}. " if i % 2 == 0 else "") + s for i, s in enumerate(san))
    ep = chess.square_name(b.ep_square) if b.ep_square is not None and b.has_legal_en_passant() else "-"
    return (f"{diagram}\n\nUppercase = White, lowercase = Black.\n"
            f"FEN: {b.fen()}\n"
            f"Side to move: {'White' if b.turn else 'Black'} (move {b.fullmove_number})\n"
            f"In check: {'yes' if b.is_check() else 'no'}\n"
            f"Castling rights: {b.castling_xfen()}\nEn passant square: {ep}\n"
            f"Move history: {hist or '(none)'}")


@mcp.tool()
def get_board() -> str:
    """The current position: ASCII diagram, FEN, side to move, check status, castling rights,
    en passant square and the full move history."""
    return describe(load())


@mcp.tool()
def legal_moves() -> str:
    """Every legal move in the current position, in SAN and UCI."""
    b = load()
    return "\n".join(f"{b.san(m)} ({m.uci()})" for m in b.legal_moves)


@mcp.tool()
def submit_move(move: str) -> str:
    """Play your move (SAN like Nf3 or UCI like g1f3). Call exactly once with your final choice."""
    out = os.environ["CHESS_OUT"]
    if os.path.exists(out):
        return "A move was already submitted this turn."
    b = load()
    try:
        m = b.parse_san(move.strip())
    except ValueError:
        try:
            m = chess.Move.from_uci(move.strip())
        except ValueError:
            return f"Could not parse '{move}'. Use SAN (e.g. Nf3) or UCI (e.g. g1f3)."
    if m not in b.legal_moves:
        return f"'{move}' is not legal here. Call legal_moves to see the options."
    open(out, "w").write(m.uci())
    return f"Accepted: {b.san(m)} ({m.uci()})."


if __name__ == "__main__":
    mcp.run()
