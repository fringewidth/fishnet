# fishnet

A chess engine compiled into the weights of a language model.

Work in progress. The target: a chess search machine built as Gated DeltaNet and MLP layers in
the circuit IR of [Qwenputer](https://github.com/divagr18/qwenputer), merged into Qwen3.5-4B,
where every step of an alpha-beta search is one forward pass (about 2 tokens per searched node),
playing against frontier LLMs that see the board through tools.

- `machine/` the spec: `ref.py` (token-level search machine, checked against recursive
  alpha-beta and python-chess move generation) and `ref2.py` (the same search with the state
  layout transformer layers can hold), plus `DESIGN.md`.
- `circuit/` the circuit in the Qwenputer IR, fp32 exact, checked token by token against the spec.
- `src/` an earlier RV32 firmware engine for the published Qwenputer checkpoint (perft-verified).
- `harness/` matches against Claude Opus 5.5 (`claude -p`) and GPT-6 Astra (`codex exec`) with
  board-inspection tools only.
