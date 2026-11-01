"""Run the chess firmware on the repo's reference RV32 machine (the model the circuit is checked
against) through one bridge call, and report trace tokens, instructions, and the response.
    measure_tokens.py ELF FEN [FEN ...]
Must run with the qwenputer riscv package importable (uv run from experimental/riscv)."""
import sys
import time

from chipformer.rv import elf, ref
from chipformer.rv import spec as S
from chipformer.rv.bridge import park


def rows_for(text: str) -> list[bytes]:
    """The text buffer as the bridge fills it: row i = the i-th token's bytes (here: 4-char
    chunks; the real Qwen tokenization gives similar row counts and identical bytes)."""
    rows = [bytes(S.TEXT_ROW)]
    for i in range(0, len(text), 4):
        rows.append(text[i:i + 4].encode().ljust(S.TEXT_ROW, b"\0"))
    rows += [bytes(S.TEXT_ROW)] * (S.TEXT_ROWS - len(rows))
    return rows


elf_path = sys.argv[1]
for fen in sys.argv[2:]:
    m = ref.Machine.from_program(elf.program(elf_path))
    park(m, max_records=10**7)
    m.text = rows_for(f"\n<function=chess>\n<parameter=fen>\n{fen}\n</parameter>\n</function>\n")
    t0 = time.time()
    toks = m.run(max_records=10**9)
    dt = time.time() - t0
    st = ref.trace_stats(toks)
    out = bytes(b for b in m.out if b < 256).decode(errors="replace")
    print({"fen": fen, **st, "secs": round(dt, 1), "fault": m.fault})
    print(repr(out))
