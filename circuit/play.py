"""Play one move with the merged checkpoint: Qwen reads the position, calls its chess engine,
the engine runs inside the same weights, and the move comes back as Qwen's own tokens.

    python play.py MODEL --fen FEN [--verify 3000] [--check-head]
    python play.py MODEL --serve 8765          (POST {"fen": ...} -> {"move": ..., ...})

Decoding is greedy and transparent to chess tokens, as the repo's rv/chat.py: a text token runs
through every layer; a chess token (PUT, START, MOVE, RET, ...) runs through the chip layers only
(Qwen's layers are muted on them, so its caches and RoPE positions see the text alone). At every
position the next token is the circuit's when its output code is set, else Qwen's argmax.

The chip layers run as the circuit itself (rv/fastexec.py's Executor: the IR's arithmetic in
fp32, the same numbers the HF modules compute, without their dense zero blocks). ``--verify N``
also runs the first N positions of the move through the HF modules of the chip layers and
requires every kept value to match the executor's bit for bit; ``--check-head`` requires the
stock lm_head's argmax over the whole vocabulary to equal the token taken at every text position.
"""
import argparse
import json
import re
import time
from pathlib import Path

import fp32  # noqa: F401
import torch

import build as B
import vocab as V
from text import Text

TOOL = {"type": "function", "function": {
    "name": "chess_engine",
    "description": "Search a chess position and return the best move in UCI notation.",
    "parameters": {"type": "object", "properties": {
        "fen": {"type": "string", "description": "the position in FEN"}}, "required": ["fen"]}}}
SYSTEM = ("You are playing a game of chess. For every position you are given, call the "
          "chess_engine tool with the position's FEN exactly as given, then reply with the move "
          "it returns.")


def exact_numerics() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


class Fishnet:
    def __init__(self, path, device: str = "cuda") -> None:
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM

        from chipformer.rv import fastexec

        exact_numerics()
        self.path = Path(path)
        self.meta = json.loads((self.path / "fishnet.json").read_text())
        self.dev = device
        self.model = Qwen3_5ForCausalLM.from_pretrained(self.path, dtype=torch.float32).to(device).eval()
        self.lm = self.model.model
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.T = Text(self.path / "tokenizer.json")
        self.c, _ = B.build_all(self.T, self.meta["depth"])
        assert self.c.stats() == self.meta["circuit"], "the circuit differs from the checkpoint's"
        self.n_chip, self.v0, self.d0 = self.meta["chip_layers"], self.meta["chip_token_offset"], \
            self.meta["chip_dims_offset"]
        self.code_cols = torch.tensor(self.meta["code_dims"], device=device)
        self.keep_cols = torch.tensor(self.meta["code_dims"] + [self.meta["tau_dim"], self.meta["mu_dim"]],
                                      device=device)
        self.keep_ir = torch.tensor(self.c.keep, device=device)
        self.ex = fastexec.Executor(self.c, device, 1)
        self.emb_cache: dict[int, torch.Tensor] = {}
        self.eos = {248046, 248044}

    # ------------------------------------------------------------------ the chip layers
    def _stream_id(self, i: int) -> int:
        return i - self.v0 if i >= self.v0 else V.TEXT0 + i

    def _emb(self, ids: list[int]) -> torch.Tensor:
        new = sorted({i for i in ids if i not in self.emb_cache})
        if new:
            for i, r in zip(new, self.ex.embed([self._stream_id(i) for i in new])):
                self.emb_cache[i] = r
        return torch.stack([self.emb_cache[i] for i in ids])

    @torch.no_grad()
    def _exec(self, i: int) -> torch.Tensor:
        """One token through the circuit; returns the kept values (code, tau, mu)."""
        x = self.ex.step(self._emb([i]))
        return x[0, self.keep_ir]

    def _hc(self, i: int, keep: torch.Tensor) -> torch.Tensor:
        """A text token's residual after the chip layers (the exit leaves Qwen's part and the
        kept dims; everything else is 0)."""
        hc = torch.zeros(self.lm.embed_tokens.weight.shape[1], device=self.dev)
        hc[: self.d0] = self.lm.embed_tokens.weight[i, : self.d0]
        hc[self.keep_cols] = keep
        return hc

    def _token_of_code(self, code: torch.Tensor) -> int | None:
        if not bool((code != 0).any()):
            return None
        t = B.decode_code((code > 0).long().tolist(), self.T)
        if t < 0:
            raise RuntimeError("the circuit's output code names no token")
        return t - V.TEXT0 if t >= V.TEXT0 else self.v0 + t

    # ------------------------------------------------------------------ Qwen's layers
    @torch.no_grad()
    def _qwen(self, hc: torch.Tensor, cache, pos: int) -> torch.Tensor:
        h = hc.view(1, 1, -1)
        p4 = torch.full((4, 1, 1), pos, device=self.dev, dtype=torch.long)
        pe = self.lm.rotary_emb(h, p4[1:])
        for i in range(self.n_chip, len(self.lm.layers)):
            h = self.lm.layers[i](h, position_embeddings=pe, attention_mask=None,
                                  position_ids=p4[0], past_key_values=cache)
        return self.model.lm_head(self.lm.norm(h))[0, -1]

    # ------------------------------------------------------------------ a move
    def prompt(self, fen: str) -> list[int]:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Position (FEN): {fen}\nYour move."}]
        out = self.tok.apply_chat_template(msgs, tools=[TOOL], add_generation_prompt=True,
                                           enable_thinking=False, tokenize=True)
        return [int(x) for x in (out if isinstance(out, list) else out["input_ids"])]

    @torch.no_grad()
    def move(self, fen: str, verify: int = 0, check_head: bool = False, max_text: int = 400,
             max_chip: int = 5_000_000, verbose: bool = False) -> dict:
        ids = self.prompt(fen)
        out = self.model(input_ids=torch.tensor([ids], device=self.dev), use_cache=True)
        cache = out.past_key_values
        # the circuit sees the prompt too (its state starts at zero: no snapshot)
        for st in self.ex.steps:
            if not isinstance(st, tuple):
                st["S"].zero_()
        self.ex.reset()
        for i in ids:
            self._exec(i)
        hf_chip = self._hf_chip_view(ids) if verify else None
        pos = len(ids)
        nxt = int(torch.argmax(out.logits[0, -1, : self.v0]))
        gen, n_text, n_chip, n_verified, head_bad = [], 0, 0, 0, 0
        t0 = time.time()
        while n_text < max_text and n_chip < max_chip:
            gen.append(nxt)
            if nxt in self.eos:
                break
            keep = self._exec(nxt)
            if hf_chip is not None and len(gen) <= verify:
                hc_hf = hf_chip(nxt)
                ref = hc_hf[self.keep_cols]
                if not torch.equal(ref, keep):
                    raise AssertionError(f"HF chip layers differ from the executor at generated "
                                         f"token {len(gen)}: {(ref != keep).nonzero().flatten().tolist()[:8]}")
                n_verified += 1
            ct = self._token_of_code(keep[: len(self.meta["code_dims"])])
            if nxt >= self.v0:  # a chess token: the circuit predicts
                n_chip += 1
                if ct is None:
                    raise RuntimeError(f"no prediction after chess token {V.decode(nxt - self.v0)}")
                nxt = ct
            else:
                n_text += 1
                logits = self._qwen(self._hc(nxt, keep), cache, pos)
                pos += 1
                pick = ct if ct is not None else int(torch.argmax(logits[: self.v0]))
                if check_head and int(torch.argmax(logits)) != pick:
                    head_bad += 1
                nxt = pick
            if verbose and len(gen) % 20000 == 0:
                print(f"  {len(gen)} tokens, {time.time() - t0:.0f}s", flush=True)
        dt = time.time() - t0
        text = self.tok.decode([i for i in gen if i < self.v0], skip_special_tokens=False)
        m = re.search(r"<tool_response>\s*([a-h][1-8][a-h][1-8][q]?)\s*</tool_response>", text)
        chess = [V.decode(i - self.v0) for i in gen if i >= self.v0]
        return {"move": m.group(1) if m else None, "text": text, "seconds": round(dt, 1),
                "chip_tokens": n_chip, "text_tokens": n_text,
                "nodes": sum(1 for t in chess if t[0] in ("MOVE", "MOVE_IMP")),
                "start": next((t for t in chess if t[0] == "START"), None),
                "puts": [t for t in chess if t[0] == "PUT"],
                "verified_positions": n_verified, "head_mismatches": head_bad if check_head else None,
                "tokens": gen}

    def _hf_chip_view(self, ids: list[int]):
        """The chip layers as HF modules on their own cache: a step function returning the
        residual after the chip layers for one token (the prompt is prefilled first)."""
        from transformers import DynamicCache

        cache = DynamicCache(config=self.model.config)
        h = self.lm.embed_tokens(torch.tensor([ids], device=self.dev))
        for L in range(self.n_chip):
            h = self.lm.layers[L](h, position_embeddings=None, attention_mask=None,
                                  position_ids=None, past_key_values=cache)

        def step(i: int) -> torch.Tensor:
            x = self.lm.embed_tokens(torch.tensor([[i]], device=self.dev))
            for L in range(self.n_chip):
                x = self.lm.layers[L](x, position_embeddings=None, attention_mask=None,
                                      position_ids=None, past_key_values=cache)
            return x[0, -1]

        return step


def serve(f: Fishnet, port: int) -> None:
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                res = f.move(req["fen"], verify=req.get("verify", 0), check_head=req.get("check_head", False))
                res.pop("tokens")
                body = json.dumps(res).encode()
                self.send_response(200)
            except Exception as e:  # noqa: BLE001
                body = json.dumps({"error": repr(e)}).encode()
                self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    print(f"serving on 127.0.0.1:{port}", flush=True)
    HTTPServer(("127.0.0.1", port), H).serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--fen", default="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    ap.add_argument("--verify", type=int, default=0)
    ap.add_argument("--check-head", action="store_true")
    ap.add_argument("--serve", type=int)
    args = ap.parse_args()
    f = Fishnet(args.model)
    if args.serve:
        serve(f, args.serve)
        return
    res = f.move(args.fen, args.verify, args.check_head, verbose=True)
    res.pop("tokens")
    res["puts"] = len(res["puts"])
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
