"""Play chess against fishnet in the browser: Qwen3.5-4B with a chess engine compiled into its
weights, running on this machine.

    uv run python app/server.py [--port 8000] [--depth 4] [--no-open]

One game at a time. On the engine's turn the page opens /think (server-sent events) and watches
the model work: Qwen's tool call as it types it, then the search tokens (every one a forward
pass), then the move it prints back.
"""
import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "circuit"))
sys.path.insert(0, os.path.join(ROOT, "machine"))
HTML = os.path.join(ROOT, "app", "index.html")


class Game:
    def __init__(self, depth: int):
        self.depth = depth
        self.board = chess.Board()
        self.human = chess.WHITE
        self.engine = None
        self.status = "loading the model (Qwen3.5-4B and the chess layers)..."
        self.lock = threading.Lock()  # one search at a time
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            from play_mac import FishnetMac
            t0 = time.time()
            self.engine = FishnetMac(self.depth)
            self.engine.ex.step(0)  # compile the circuit's step once (numba)
            self.engine.ex.reset()
            self.status = f"ready (loaded in {time.time() - t0:.0f}s on {self.engine.dev})"
        except Exception as e:  # noqa: BLE001
            self.status = f"failed to load: {e!r}"

    def state(self) -> dict:
        b = self.board
        out = b.outcome(claim_draw=True)
        sans, tmp = [], chess.Board()
        for m in b.move_stack:
            sans.append(tmp.san(m))
            tmp.push(m)
        return {"fen": b.fen(), "legal": [m.uci() for m in b.legal_moves], "history": sans,
                "last": b.move_stack[-1].uci() if b.move_stack else None,
                "human": "white" if self.human == chess.WHITE else "black",
                "turn": "white" if b.turn == chess.WHITE else "black",
                "result": out.result() if out else None,
                "termination": out.termination.name.lower() if out else None,
                "ready": self.engine is not None, "status": self.status}


def reference_move(fen: str, depth: int) -> str:
    """The spec machine's move (machine/ref.py), which the weights must reproduce exactly."""
    import ref
    pos, white = ref.from_fen(fen)
    m = ref.Machine(pos, depth)
    done = m.run()
    b = chess.Board(fen)
    f, t = ref.real_square(done[1], white), ref.real_square(done[2], white)
    promo = chess.QUEEN if b.piece_type_at(f) == chess.PAWN and chess.square_rank(t) in (0, 7) else None
    return chess.Move(f, t, promotion=promo).uci()


def make_handler(game: Game):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):  # noqa: N802
            if self.path in ("/", "/index.html"):
                body = open(HTML, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/state":
                self._json(game.state())
            elif self.path.startswith("/think"):
                self._think("verify=1" in self.path)
            else:
                self.send_error(404)

        def do_POST(self):  # noqa: N802
            req = self._body()
            if self.path == "/new":
                with game.lock:
                    game.board = chess.Board()
                    game.human = chess.WHITE if req.get("color", "white") == "white" else chess.BLACK
                self._json(game.state())
            elif self.path == "/move":
                with game.lock:
                    try:
                        m = chess.Move.from_uci(req["uci"])
                    except (KeyError, ValueError):
                        return self._json({"error": "bad move"}, 400)
                    if game.board.turn != game.human or m not in game.board.legal_moves:
                        return self._json({"error": "illegal move"}, 400)
                    game.board.push(m)
                self._json(game.state())
            else:
                self.send_error(404)

        def _think(self, verify: bool):
            """The engine's move, streamed: Qwen's text, the search's progress, the result."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def send(ev):
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                self.wfile.flush()

            if game.engine is None:
                return send({"error": game.status})
            if not game.lock.acquire(blocking=False):
                return send({"error": "already thinking"})
            try:
                b = game.board
                if b.turn == game.human or b.is_game_over(claim_draw=True):
                    return send({"error": "not the engine's turn"})
                fen = b.fen()
                import vocab as V
                eng = game.engine
                st = {"chess": 0, "nodes": 0, "last": 0.0, "recent": deque(maxlen=16), "phase": "call"}

                def on_token(i, is_chess):
                    now = time.time()
                    if is_chess:
                        st["chess"] += 1
                        tok = V.decode(i - eng.v0)
                        if tok[0] in ("MOVE", "MOVE_IMP"):
                            st["nodes"] += 1
                        st["recent"].append(V.label(tok))
                        st["phase"] = "search"
                        if now - st["last"] > 0.15:
                            st["last"] = now
                            send({"phase": "search", "tokens": st["chess"], "nodes": st["nodes"],
                                  "recent": list(st["recent"])})
                    else:
                        if st["phase"] == "search":
                            send({"phase": "search", "tokens": st["chess"], "nodes": st["nodes"],
                                  "recent": list(st["recent"])})
                            st["phase"] = "reply"
                        send({"phase": st["phase"], "text": eng.T.decode([i])})

                res = eng.move(fen, on_token=on_token)
                uci = res["move"]
                move = chess.Move.from_uci(uci) if uci else None
                if move is None or move not in b.legal_moves:
                    return send({"error": f"the model did not produce a legal move ({uci!r})"})
                ok = None
                if verify:
                    send({"phase": "verify"})
                    ok = reference_move(fen, eng.depth) == uci
                san = b.san(move)
                b.push(move)
                send({"done": True, "move": uci, "san": san, "verified": ok,
                      "stats": {k: res[k] for k in ("seconds", "chip_tokens", "nodes", "text_tokens")},
                      "state": game.state()})
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                game.lock.release()

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    game = Game(args.depth)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(game))
    url = f"http://127.0.0.1:{args.port}"
    print(f"fishnet: {url}  (the model loads in the background; the page shows when it is ready)", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    srv.serve_forever()


if __name__ == "__main__":
    main()
