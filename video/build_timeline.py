"""Replay timeline for the video, from a game's records.

    uv run python video/build_timeline.py TAG [--pace tweet|full] > timeline.json

Reads harness/records/TAG.jsonl (one line per ply) and harness/logs/TAG/ply###-fishnet.json.gz
(Qwen's whole token stream per move). Each ply becomes a segment of video time; `pace` decides how
long: "full" gives every ply the same screen time, "tweet" keeps the opening and the finish at
full pace and compresses the middle.
"""
import argparse
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "circuit"))
sys.path.insert(0, os.path.join(ROOT, "machine"))

import vocab as V  # noqa: E402
from text import Text  # noqa: E402

def qwen_ply(path: str, T: Text) -> dict:
    with gzip.open(path, "rt") as fh:
        res = json.load(fh)
    v0 = res["chip_token_offset"]
    toks = res["tokens"]
    first_chip = next((i for i, t in enumerate(toks) if t >= v0), len(toks))
    last_chip = max((i for i, t in enumerate(toks) if t >= v0), default=first_chip)
    call = T.decode([t for t in toks[:first_chip] if t < v0])
    reply = T.decode([t for t in toks[last_chip + 1:] if t < v0])
    labels = [V.label(V.decode(t - v0)) for t in toks[first_chip:last_chip + 1]]
    n = len(labels)
    head, tail = labels[:80], labels[-40:] if n > 120 else []
    mid = labels[80:n - 40] if n > 120 else labels[80:]
    step = max(1, len(mid) // 240)
    stream = head + mid[::step][:240] + tail
    return {"call": call.replace("<|im_end|>", ""), "reply": reply.replace("<|im_end|>", "").strip(),
            "stream": stream, "chip_tokens": res["chip_tokens"], "nodes": res["nodes"],
            "seconds": res["seconds"], "move": res["move"]}


def opus_ply(detail: dict) -> dict:
    ev = []
    for e in detail.get("events", []):
        if e["kind"] == "text" and e["text"].strip():
            ev.append({"kind": "text", "text": e["text"].strip()})
        elif e["kind"] == "tool_use":
            name = e["name"].replace("mcp__chess__", "")
            arg = e.get("input") or {}
            ev.append({"kind": "tool", "text": f"{name}({', '.join(f'{k}={v}' for k, v in arg.items())})"})
    return {"events": ev, "seconds": detail.get("seconds")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--pace", default="tweet", choices=["tweet", "full"])
    args = ap.parse_args()
    T = Text()
    recs = [json.loads(line) for line in open(os.path.join(ROOT, "harness", "records", f"{args.tag}.jsonl"))]
    plies, t = [], 2.5  # an intro card first
    n = len(recs)
    for i, r in enumerate(recs):
        qwen = r["player"].startswith("fishnet")
        if qwen:
            p = os.path.join(ROOT, "harness", "logs", args.tag, f"ply{r['ply'] - 1:03d}-fishnet.json.gz")
            body = qwen_ply(p, T)
        else:
            body = opus_ply(r["detail"] or {})
        if args.pace == "full":
            dur = 4.0 if qwen else 3.0
        else:  # full pace for the first 8 and last 10 plies, fast in between
            edge = i < 8 or i >= n - 10
            dur = (4.0 if qwen else 3.0) if edge else (1.1 if qwen else 0.7)
        plies.append({"ply": r["ply"], "side": "qwen" if qwen else "opus", "san": r["san"], "uci": r["uci"],
                      "fen_before": r["fen_before"], "fen_after": r["fen_after"],
                      "real_seconds": round(r["t_end"] - r["t_start"], 1), "t0": round(t, 3),
                      "dur": dur, **body})
        t += dur
    import chess
    b = chess.Board(recs[-1]["fen_after"])
    outcome = b.outcome(claim_draw=True)
    result = {"result": b.result(claim_draw=True),
              "termination": outcome.termination.name.lower() if outcome else "unfinished"}
    json.dump({"tag": args.tag, "plies": plies, "end": round(t + 5.0, 3), **result}, sys.stdout)


if __name__ == "__main__":
    main()
