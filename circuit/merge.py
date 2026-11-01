"""Merge the chess machine into Qwen3.5-4B: one stock ``Qwen3_5ForCausalLM`` checkpoint.

    python merge.py SRC --out DST [--depth 4] [--no-mute]
    python merge.py --chip-only [--depth 4]   (emit and check the chip layers, write nothing)

Layout (the repo's "v2", as rv/merge.py):

- the chip layers come first (the circuit of build.build_all), then Qwen's 32 layers, zero-padded
  to the new hidden size (Qwen never reads or writes chip dims; its full-residual norms are
  rescaled so its normalized inputs are unchanged);
- every Qwen vocabulary row carries the ballast and its FEN features in the chip dims; the chess
  tokens (START, MOVE, RET, PUT, ...) are appended after Qwen's vocabulary as special tokens;
- the tied output head reads only the output code and tau among the chip dims, and Qwen's layers
  are muted on chess tokens (q35.mute, on mu).

The chip layers are fp32: their weights (PeSTO values, 16-bit scores) are not bf16 values, and
their arithmetic is exact in fp32 (circuit/spans.py checks the summation spans). Qwen's tensors
are stored in bf16 as shipped; load the checkpoint with dtype=torch.float32 (bf16 -> fp32 is
exact). ``fishnet.json`` records the layout for the decoder (play.py).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import fp32  # noqa: F401  (fp32 IR checks, fp32 emitter)
import numpy as np
import torch

import build as B
import vocab as V
from text import Text

SRC_TEXT = "model.language_model."
DST = "model."


def chip_token_names() -> list[str]:
    out = []
    for t in range(V.VOCAB):
        d = V.decode(t)
        out.append("<|fn:" + ":".join(str(x) for x in d) + "|>")
    return out


def extend_tokenizer(src: Path, dst: Path, first_id: int) -> None:
    import shutil

    from tokenizers import AddedToken, Tokenizer

    tok = Tokenizer.from_file(str(src / "tokenizer.json"))
    n = tok.get_vocab_size(with_added_tokens=True)
    pad = [AddedToken(f"<|reserved_{i}|>", special=True) for i in range(first_id - n)]
    chip = [AddedToken(s, special=True) for s in chip_token_names()]
    tok.add_special_tokens(pad + chip)
    assert tok.token_to_id(chip_token_names()[0]) == first_id
    dst.mkdir(parents=True, exist_ok=True)
    tok.save(str(dst / "tokenizer.json"))
    for f in ("chat_template.jinja", "LICENSE"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    tc = json.loads((src / "tokenizer_config.json").read_text(encoding="utf-8"))
    dec = tc.setdefault("added_tokens_decoder", {})
    for i, t in enumerate(pad + chip):
        dec[str(n + i)] = {"content": t.content, "lstrip": False, "normalized": False,
                           "rstrip": False, "single_word": False, "special": True}
    (dst / "tokenizer_config.json").write_text(json.dumps(tc, indent=2, ensure_ascii=False),
                                               encoding="utf-8")


def feature_rows(c, lay, toks: list[int], feats: list[int]) -> np.ndarray:
    """(len(toks), len(feats)) feature values."""
    out = np.zeros((len(toks), len(feats)), dtype=np.float64)
    for j, d in enumerate(feats):
        out[:, j] = [float(c.feature_value(d, t)) for t in toks]
    return out


def chip_rows(c, lay) -> torch.Tensor:
    """Rows of every chess token (chip dims only; the Qwen part is 0)."""
    import functools

    V.decode = functools.lru_cache(maxsize=None)(V.decode)  # every feature decodes its token
    feats = sorted(c.token_fns)
    cols = [lay.dim(d) for d in feats]
    e = torch.zeros(V.VOCAB, lay.hidden, dtype=torch.float32)
    step = 8192
    for a in range(0, V.VOCAB, step):
        toks = list(range(a, min(a + step, V.VOCAB)))
        e[a:a + len(toks), cols] = torch.tensor(feature_rows(c, lay, toks, feats), dtype=torch.float32)
        print(f"  chip rows {a + len(toks)}/{V.VOCAB}", flush=True)
    e[:, lay.off:lay.off + lay.m] = lay.ballast
    return e


def text_rows(c, lay, T: Text, base_emb: torch.Tensor, v0: int) -> torch.Tensor:
    """Qwen's rows: its own part, plus the chip features of each text token (computed for the
    features any text token can have; the rest are checked to be 0 on a sample)."""
    feats = sorted(c.token_fns)
    rng = np.random.default_rng(0)
    special = sorted(set(T.fen_info) | {T.TCO, T.TCC} | set(T.print_ids.values()))
    sample = sorted(set(rng.choice(special, 3000, replace=False).tolist())
                    | set(rng.integers(0, v0, 3000).tolist()) | {T.TCO, T.TCC} | set(T.print_ids.values()))
    vals = feature_rows(c, lay, [V.TEXT0 + q for q in sample], feats)
    tfeats = [d for j, d in enumerate(feats) if np.any(vals[:, j] != vals[0, j]) or vals[0, j] != 0]
    print(f"  text features: {len(tfeats)} of {len(feats)}: "
          f"{sorted(c.dims[d].name for d in tfeats)[:12]}...", flush=True)
    h = lay.hidden
    d0 = base_emb.shape[1]
    full = torch.zeros(v0, h, dtype=torch.float32)
    full[:, :d0] = base_emb[:v0, :d0].float()
    cols = [lay.dim(d) for d in tfeats]
    step = 16384
    for a in range(0, v0, step):
        qs = list(range(a, min(a + step, v0)))
        full[a:a + len(qs)][:, cols] = torch.tensor(
            feature_rows(c, lay, [V.TEXT0 + q for q in qs], tfeats), dtype=torch.float32)
    full[:, lay.off:lay.off + lay.m] = lay.ballast
    # check: the full feature vector of sampled tokens equals the row
    allcols = [lay.dim(d) for d in feats]
    for q, row in zip(sample[:500], vals[:500]):
        assert torch.equal(full[q, allcols], torch.tensor(row, dtype=torch.float32)), q
    return full


def build_circuit(T: Text, depth: int, tc0: dict | None = None):
    from chipformer.q35.ir import Target

    target = B.T4B if tc0 is None else Target(tc0["linear_num_key_heads"], tc0["linear_num_value_heads"],
                                               tc0["intermediate_size"])
    c, parts = B.build_all(T, depth) if target == B.T4B else (None, None)
    assert c is not None, "Qwen3.5-4B's layer shapes expected"
    return c


def layout(c, d0: int):
    from chipformer.q35.emit import Layout, allocate
    from chipformer.q35.merge import pick_hidden

    keep = set(c.keep)
    alloc = allocate(c, None, keep, c.target.intermediate)
    hidden, m = pick_hidden(d0, 1 + alloc.n_slots)
    lay = Layout(hidden=hidden, off=d0, m=m, slots=alloc.slots)
    lay.scale  # noqa: B018 - asserts m * B^2 / hidden is a power of 4
    return lay, alloc, keep


def chip_only(depth: int) -> None:
    """Emit the chip layers one at a time (fp32 checks) without Qwen's weights."""
    from chipformer.q35.emit import emit_layers

    T = Text()
    c = build_circuit(T, depth)
    lay, alloc, keep = layout(c, 2560)
    tc = {"hidden_size": lay.hidden, "linear_num_key_heads": 16, "linear_num_value_heads": 32,
          "intermediate_size": 9216, "linear_conv_kernel_dim": 4}
    print(f"circuit {c.stats()}, slots {alloc.n_slots}, hidden {lay.hidden} (m={lay.m})", flush=True)
    sd = emit_layers(c, lay, tc, 0, DST, alloc, keep)
    n = sum(t.numel() for t in sd.values())
    nz = sum(int((t != 0).sum()) for t in sd.values())
    print(f"chip layers: {n / 1e9:.2f}B parameters ({nz / 1e6:.1f}M nonzero), all fp32-exact")


def build_merged(src, dst, depth: int, mute_qwen: bool = True) -> dict:
    from safetensors import safe_open
    from safetensors.torch import save_file

    from chipformer.q35.emit import emit_layers
    from chipformer.q35.merge import IM_START, IN_COLS, OUT_ROWS, TAU_GAIN, WIDE_NORMS
    from chipformer.q35.tiny import code_gain

    src, dst = Path(src), Path(dst)
    cfg = json.loads((src / "config.json").read_text())
    tc0 = cfg["text_config"]
    d0, v0 = tc0["hidden_size"], tc0["vocab_size"]
    T = Text(src / "tokenizer.json")
    c = build_circuit(T, depth, tc0)
    lay, alloc, keep = layout(c, d0)
    hidden = lay.hidden
    n_chip, n0 = c.n_layers(), tc0["num_hidden_layers"]
    print(f"circuit {c.stats()}, slots {alloc.n_slots}, hidden {hidden} (m={lay.m}), "
          f"{n_chip} chip layers", flush=True)

    tc = {k: v for k, v in tc0.items() if k != "full_attention_interval"}
    tc.update(hidden_size=hidden, num_hidden_layers=n_chip + n0, mtp_num_hidden_layers=0,
              layer_types=["linear_attention"] * n_chip + tc0["layer_types"],
              vocab_size=-(-(v0 + V.VOCAB) // 64) * 64, tie_word_embeddings=True,
              architectures=["Qwen3_5ForCausalLM"], model_type="qwen3_5_text", dtype="float32")
    vocab = tc["vocab_size"]

    sinks = None
    if mute_qwen:
        from chipformer.q35.mute import fit_anti_sink

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        sinks = fit_anti_sink(src, dev)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    scale = (d0 / hidden) ** 0.5
    sd = {}
    base_emb = None
    for f in sorted(src.glob("*.safetensors")):
        with safe_open(f, "pt") as fh:
            for k in fh.keys():  # noqa: SIM118
                if not k.startswith(SRC_TEXT):
                    continue  # vision tower, mtp
                w = fh.get_tensor(k)
                if k == SRC_TEXT + "embed_tokens.weight":
                    base_emb = w
                    continue
                if IN_COLS.search(k):
                    w = torch.cat([w, w.new_zeros(*w.shape[:-1], hidden - d0)], -1)
                elif OUT_ROWS.search(k):
                    w = torch.cat([w, w.new_zeros(hidden - d0, *w.shape[1:])], 0)
                elif WIDE_NORMS.search(k):
                    g = (1.0 + w.float()) * scale - 1.0
                    w = torch.cat([g.to(w.dtype), w.new_zeros(hidden - d0)])
                k2 = DST + k[len(SRC_TEXT):]
                k2 = re.sub(r"^(model\.layers\.)(\d+)\.", lambda mt: f"{mt[1]}{int(mt[2]) + n_chip}.", k2)
                sd[k2] = w
    assert base_emb is not None
    if mute_qwen:
        from chipformer.q35.mute import mute

        mute(sd, tc, n0, lay.dim(next(iter(c.mu.terms))), hidden**0.5, prefix=DST, sinks=sinks,
             first=n_chip)

    layers = emit_layers(c, lay, tc, 0, DST, alloc, keep)  # fp32
    sd.update(layers)
    del layers

    emb = torch.zeros(vocab, hidden, dtype=torch.float32)
    emb[:v0] = text_rows(c, lay, T, base_emb, v0)
    del base_emb
    emb[v0:v0 + V.VOCAB] = chip_rows(c, lay)
    sd[DST + "embed_tokens.weight"] = emb

    code = [next(iter(e.terms)) for e in c.code]
    tau = next(iter(c.tau.terms))
    norm = sd[DST + "norm.weight"].float()
    norm[d0:] = -1.0  # chip dims are invisible to the lm_head ...
    for d in code:  # ... except the output code and tau
        norm[lay.dim(d)] = code_gain(lay)
    norm[lay.dim(tau)] = TAU_GAIN
    sd[DST + "norm.weight"] = norm

    dst.mkdir(parents=True, exist_ok=True)
    shard, size, shards = {}, 0, []
    for k in sorted(sd):
        shard[k] = sd[k].contiguous()
        size += sd[k].numel() * sd[k].element_size()
        if size > 4 * 2**30:
            shards.append(shard)
            shard, size = {}, 0
    shards.append(shard)
    index = {"metadata": {"total_size": sum(t.numel() * t.element_size() for t in sd.values())},
             "weight_map": {}}
    for i, sh in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(sh, str(dst / name), metadata={"format": "pt"})
        index["weight_map"].update({k: name for k in sh})
        print(f"  wrote {name}", flush=True)
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    (dst / "config.json").write_text(json.dumps(tc, indent=2))
    extend_tokenizer(src, dst, v0)
    gen = {"eos_token_id": [248046, tc0.get("eos_token_id", 248044)],
           "pad_token_id": tc0.get("eos_token_id", 248044), "bad_words_ids": [[IM_START]],
           "do_sample": False}
    (dst / "generation_config.json").write_text(json.dumps(gen, indent=2))
    meta = {"layout": "fishnet1", "chip_token_offset": v0, "chip_vocab": V.VOCAB, "depth": depth,
            "chip_layers": n_chip, "qwen_layers_from": n_chip, "hidden": hidden,
            "chip_dims_offset": d0, "ballast_dims": lay.m, "circuit": c.stats(),
            "slots": alloc.n_slots, "muted": mute_qwen,
            "code_dims": [lay.dim(d) for d in code], "tau_dim": lay.dim(tau),
            "mu_dim": lay.dim(next(iter(c.mu.terms))), "nfam": B.NFAM, "npay": B.NPAY,
            "tool_call": [T.TCO, T.TCC]}
    (dst / "fishnet.json").write_text(json.dumps(meta, indent=1))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", nargs="?")
    ap.add_argument("--out", default="dist/fishnet-q4")
    ap.add_argument("--depth", type=int, default=B.DEPTH)
    ap.add_argument("--no-mute", action="store_true")
    ap.add_argument("--chip-only", action="store_true")
    args = ap.parse_args()
    if args.chip_only:
        chip_only(args.depth)
        return
    if not args.src:
        sys.exit("SRC (a Qwen3.5-4B snapshot directory) is required")
    meta = build_merged(args.src, args.out, args.depth, not args.no_mute)
    print(json.dumps({k: meta[k] for k in ("hidden", "chip_layers", "slots", "circuit")}))


if __name__ == "__main__":
    main()
