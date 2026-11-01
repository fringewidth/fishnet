"""Qwen's vocabulary as the bridge sees it: each text token's bytes, read as FEN.

The circuit reads a FEN that Qwen writes inside a tool call, one Qwen token at a time. A token's
FEN features depend on its bytes alone, so they are token features (in the merged model: columns
of that token's embedding row):

  board-clean   every byte is a piece letter, a digit 1-8 or '/' (a token of the board field)
  width         squares it covers in FEN order (a letter 1, a digit d, '/' 0), 0..8
  code[k]       the piece code on its k-th square (0 empty; 1-6 White PNBRQK, 7-12 Black)
  fen-clean     every byte is one a FEN can hold (board bytes, space, w, a-h, '-', digits)
  space, newline, and the letters of the side / castling / en passant fields
"""
import glob
import os
from functools import cached_property
from pathlib import Path

BOARD_BYTES = set(b"pnbrqkPNBRQK12345678/")
FEN_BYTES = BOARD_BYTES | set(b" wabcdefgh-0123456789")
PIECE = {ord(ch): i + 1 for i, ch in enumerate("PNBRQK")} | {ord(ch): i + 7 for i, ch in enumerate("pnbrqk")}
MAX_W = 8

# the text tokens the circuit prints (its echo of the result); index = the Q code's payload.
# Files at 8 + x and ranks at 16 + y, so their payload bits are the square's bits.
PRINT = {0: "<tool_response>", 1: "\n", 2: "q", 3: "</tool_response>"}
PRINT.update({8 + x: "abcdefgh"[x] for x in range(8)})
PRINT.update({16 + y: "12345678"[y] for y in range(8)})


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte-level BPE alphabet: byte -> the character that stands for it."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def default_tokenizer() -> Path:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    hits = sorted(glob.glob(str(hub / "models--Qwen--Qwen3.5-4B" / "snapshots" / "*" / "tokenizer.json")))
    if not hits:
        raise FileNotFoundError("no Qwen3.5-4B tokenizer.json in the HF cache")
    return Path(hits[-1])


class Text:
    def __init__(self, tokenizer_json: str | Path | None = None) -> None:
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(tokenizer_json or default_tokenizer()))
        vocab = self.tok.get_vocab(with_added_tokens=True)
        self.n = max(vocab.values()) + 1
        dec = {c: b for b, c in bytes_to_unicode().items()}
        added = {a.content for a in self.tok.get_added_tokens_decoder().values()}
        self.bytes = [b""] * self.n
        for s, i in vocab.items():
            if s in added or not all(ch in dec for ch in s):
                self.bytes[i] = s.encode()
            else:
                self.bytes[i] = bytes(dec[ch] for ch in s)
        self.added = {vocab[s] for s in added}
        self.TCO, self.TCC = vocab["<tool_call>"], vocab["</tool_call>"]
        u = bytes_to_unicode()
        self.print_ids = {}
        for p, s in PRINT.items():
            self.print_ids[p] = vocab[s] if s in vocab else vocab["".join(u[b] for b in s.encode())]
        self.print_index = {q: p for p, q in self.print_ids.items()}

    def encode(self, s: str) -> list[int]:
        return self.tok.encode(s, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.bytes[i] for i in ids).decode("utf-8", "replace")

    @cached_property
    def fen_info(self) -> dict:
        """qwen id -> its FEN features, for the tokens that have any (others: all zero)."""
        out = {}
        for q, b in enumerate(self.bytes):
            if q in self.added or not b:
                continue
            info = {"nl": int(10 in b), "space": int(32 in b), "fenok": int(set(b) <= FEN_BYTES)}
            board_ok = set(b) <= BOARD_BYTES
            codes = []
            if board_ok:
                for ch in b:
                    if ch in PIECE:
                        codes.append(PIECE[ch])
                    elif 49 <= ch <= 56:
                        codes += [0] * (ch - 48)
                if len(codes) > MAX_W:
                    board_ok = False
            info["boardok"] = int(board_ok)
            info["codes"] = codes if board_ok else []
            letters = set(b) if info["fenok"] else set()
            for ch in "wbKQkq":
                info["has_" + ch] = int(ord(ch) in letters)
            files = [x for x in range(8) if ord("abcdefgh"[x]) in letters]
            info["file"] = files[0] if len(files) == 1 else None
            if any(v for k, v in info.items() if k not in ("codes", "file")) or info["codes"] or \
                    info["file"] is not None:
                out[q] = info
        return out
