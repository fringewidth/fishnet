"""One game against Claude Opus 5.5, which sees the board through tools and nothing else.

    uv run python harness/match.py PLAYER COLOR TAG [--resume]

PLAYER is `fishnet` (the model, served by `circuit/play_mac.py --serve 8765`) or `dN` (the
reference search at depth N). COLOR is the colour PLAYER plays. Writes, under harness/:
records/TAG.jsonl (one line per ply), games/TAG.pgn, and logs/TAG/ (Opus's event streams and
the model's token streams). --resume continues a game from records/TAG.jsonl.
"""
import datetime
import json
import os
import sys
import time

import chess
import chess.pgn

from claude_player import ClaudePlayer
from players import Fishnet, Reference

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_PLY = 300


def main():
    spec, color, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    for d in ("games", "records", f"logs/{tag}"):
        os.makedirs(os.path.join(HERE, d), exist_ok=True)
    log_dir = os.path.join(HERE, "logs", tag)
    rec_path = os.path.join(HERE, "records", f"{tag}.jsonl")
    engine = Fishnet(log_dir=log_dir) if spec == "fishnet" else Reference(int(spec[1:]))
    opus = ClaudePlayer(model="claude-opus-5-5", effort="high", log_dir=log_dir)
    white, black = (engine, opus) if color == "white" else (opus, engine)

    b = chess.Board()
    notes = []
    if "--resume" in sys.argv and os.path.exists(rec_path):
        for line in open(rec_path):
            r = json.loads(line)
            b.push_uci(r["uci"])
            notes.append(r["note"])
        print(f"{tag} resumed at ply {b.ply()}", flush=True)
    elif os.path.exists(rec_path):
        os.remove(rec_path)

    with open(rec_path, "a") as rec:
        while not b.is_game_over(claim_draw=True) and b.ply() < MAX_PLY:
            p = white if b.turn == chess.WHITE else black
            fen_before, t_start = b.fen(), time.time()
            p.detail = None
            m = p.move(b)
            t_end = time.time()
            assert m in b.legal_moves, (p.name, m.uci(), b.fen())
            san = b.san(m)
            notes.append(p.last)
            b.push(m)
            rec.write(json.dumps({"ply": b.ply(), "player": p.name, "fen_before": fen_before, "uci": m.uci(),
                                  "san": san, "fen_after": b.fen(), "t_start": t_start, "t_end": t_end,
                                  "note": p.last, "detail": p.detail}) + "\n")
            rec.flush()
            print(f"{tag} ply {b.ply():3d} {p.name:32s} {san:8s} {p.last}", flush=True)

    result = b.result(claim_draw=True)
    g = chess.pgn.Game.from_board(b)
    g.headers.update({"Event": f"{engine.name} vs {opus.name}", "White": white.name, "Black": black.name,
                      "Result": result, "Date": datetime.date.today().isoformat()})
    node = g
    for n in notes:
        node = node.variations[0]
        node.comment = n
    open(os.path.join(HERE, "games", f"{tag}.pgn"), "w").write(str(g) + "\n")
    outcome = b.outcome(claim_draw=True)
    print(f"RESULT {tag}: {result} ({outcome.termination.name if outcome else 'ply cap'})", flush=True)
    engine.close()


if __name__ == "__main__":
    main()
