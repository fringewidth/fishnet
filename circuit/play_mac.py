"""Play one move on a Mac: Qwen3.5-4B as shipped (MPS, bf16) plus the chess circuit's layers on
the CPU (sparse_exec, fp32, exact).

    python play_mac.py --fen FEN
    python play_mac.py --serve 8765      (POST {"fen": ...}, the same API as play.py)

This computes what the merged checkpoint computes (merge.py: the chip layers first, then Qwen's
layers zero-padded), without materializing its zero blocks:

- every token, text or chess, goes through the chip layers (the circuit reads Qwen's text);
- a text token then goes through Qwen's own layers (in the merged model they see exactly Qwen's
  embedding part: the chip layers' exit clears everything else but the tiny output code / tau);
- chess tokens skip Qwen's layers (transparent decoding, as the repo's rv/chat.py: in the merged
  model Qwen is muted on them);
- the next token is the circuit's where its output code is set (it owns the position), else
  Qwen's greedy argmax.
"""
import argparse
import json
import re
import time

import fp32  # noqa: F401
import numpy as np
import torch

import build as B
import vocab as V
from play import SYSTEM, TOOL
from sparse_exec import SparseExec
from text import Text, default_tokenizer


class FishnetMac:
    def __init__(self, depth: int = B.DEPTH, device: str = "mps") -> None:
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        src = default_tokenizer().parent
        self.dev = device
        self.tok = AutoTokenizer.from_pretrained(src)
        full = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16)
        self.lm = full.model.language_model.to(device).eval()
        self.head = full.lm_head.to(device).eval()
        del full
        self.v0 = self.head.weight.shape[0]
        self.T = Text(src / "tokenizer.json")
        self.c, _ = B.build_all(self.T, depth)
        self.ex = SparseExec(self.c)
        self.code_dims = np.array([next(iter(e.terms)) for e in self.c.code])
        self.eos = {248046, 248044}
        self.banned = [248045]  # <|im_start|>: banned in the merged checkpoint's generation config
        self.depth = depth

    def _sid(self, i: int) -> int:
        return i - self.v0 if i >= self.v0 else V.TEXT0 + i

    def _circuit(self, i: int) -> int | None:
        """One token through the chip layers; the model id the circuit predicts, or None."""
        x = self.ex.step(self._sid(i))
        code = x[self.code_dims]
        if not np.any(code != 0):
            return None
        t = B.decode_code([int(v > 0) for v in code], self.T)
        if t < 0:
            raise RuntimeError("the circuit's output code names no token")
        return t - V.TEXT0 if t >= V.TEXT0 else self.v0 + t

    @torch.no_grad()
    def _qwen(self, ids: list[int], cache):
        out = self.lm(input_ids=torch.tensor([ids], device=self.dev), past_key_values=cache, use_cache=True)
        logits = self.head(out.last_hidden_state[:, -1])[0].float()
        logits[self.banned] = -float("inf")
        return logits, out.past_key_values

    def prompt(self, fen: str) -> list[int]:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Position (FEN): {fen}\nYour move."}]
        out = self.tok.apply_chat_template(msgs, tools=[TOOL], add_generation_prompt=True,
                                           enable_thinking=False, tokenize=True)
        return [int(x) for x in (out if isinstance(out, list) else out["input_ids"])]

    @torch.no_grad()
    def move(self, fen: str, max_text: int = 600, max_chip: int = 20_000_000, verbose=False, **_) -> dict:
        ids = self.prompt(fen)
        self.ex.reset()  # the circuit starts from zero state and reads the prompt
        for i in ids:
            self._circuit(i)
        logits, cache = self._qwen(ids, None)
        nxt = int(torch.argmax(logits[: self.v0]))
        gen, times, n_text, n_chip = [], [], 0, 0
        t0 = time.time()
        while n_text < max_text and n_chip < max_chip:
            gen.append(nxt)
            if nxt < self.v0 or n_chip % 1000 == 0:
                times.append((len(gen) - 1, round(time.time() - t0, 3)))
            if nxt in self.eos:
                break
            ct = self._circuit(nxt)
            if nxt >= self.v0:
                n_chip += 1
                if ct is None:
                    raise RuntimeError(f"no prediction after chess token {V.decode(nxt - self.v0)}")
                nxt = ct
            else:
                n_text += 1
                logits, cache = self._qwen([nxt], cache)
                nxt = ct if ct is not None else int(torch.argmax(logits[: self.v0]))
            if verbose and n_chip and n_chip % 100000 == 0:
                print(f"  {n_chip} chess tokens, {time.time() - t0:.0f}s", flush=True)
        dt = time.time() - t0
        text = self.tok.decode([i for i in gen if i < self.v0], skip_special_tokens=False)
        m = re.search(r"<tool_response>\s*([a-h][1-8][a-h][1-8]q?)\s*</tool_response>", text)
        chess = [V.decode(i - self.v0) for i in gen if i >= self.v0]
        return {"move": m.group(1) if m else None, "text": text, "seconds": round(dt, 1),
                "chip_tokens": n_chip, "text_tokens": n_text,
                "nodes": sum(1 for t in chess if t[0] in ("MOVE", "MOVE_IMP")),
                "start": next((t for t in chess if t[0] == "START"), None),
                "puts": [t for t in chess if t[0] == "PUT"],
                "prompt": self.tok.decode(ids, skip_special_tokens=False), "times": times,
                "chip_token_offset": self.v0, "tokens": gen}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", default="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    ap.add_argument("--depth", type=int, default=B.DEPTH)
    ap.add_argument("--serve", type=int)
    args = ap.parse_args()
    f = FishnetMac(args.depth)
    if args.serve:
        from play import serve
        serve(f, args.serve)
        return
    res = f.move(args.fen, verbose=True)
    res.pop("tokens")
    res["times"] = res["times"][-3:]
    res["puts"] = len(res["puts"])
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
